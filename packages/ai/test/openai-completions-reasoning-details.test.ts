import { Type } from "typebox";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { stream as streamOpenAICompletions } from "../src/api/openai-completions.ts";
import type { AssistantMessage, Model, Tool } from "../src/types.ts";
import { normalizeContext } from "../src/utils/transcript.ts";

const mockState = vi.hoisted(() => ({
	chunkSets: [] as unknown[][],
	payloads: [] as unknown[],
}));

vi.mock("openai", () => {
	class FakeOpenAI {
		chat = {
			completions: {
				create: (payload: unknown) => {
					mockState.payloads.push(payload);
					const chunks = mockState.chunkSets.shift() ?? [];
					const stream = {
						async *[Symbol.asyncIterator]() {
							for (const chunk of chunks) {
								yield chunk;
							}
						},
					};
					const result = Promise.resolve(stream) as Promise<typeof stream> & {
						withResponse: () => Promise<{ data: typeof stream; response: { status: number; headers: Headers } }>;
					};
					result.withResponse = async () => ({
						data: stream,
						response: { status: 200, headers: new Headers() },
					});
					return result;
				},
			},
		};
	}
	return { default: FakeOpenAI };
});

const reasoningDetail = { type: "reasoning.encrypted", id: "call_1", data: "encrypted-signature" };
const signedReasoningTextDetail = {
	type: "reasoning.text",
	text: "I should call the read tool.",
	signature: "sha256:signed-text",
	id: "reasoning-text-1",
	format: "anthropic-claude-v1",
	index: 0,
};
const reasoningSummaryDetail = {
	type: "reasoning.summary",
	summary: "Decided to inspect the requested file.",
	id: "reasoning-summary-1",
	format: "anthropic-claude-v1",
	index: 1,
};
const readTool: Tool = {
	name: "read",
	description: "Read a file",
	parameters: Type.Object({ path: Type.String() }),
};

function model(): Model<"openai-completions"> {
	return {
		id: "google/gemini-test",
		name: "Gemini Test",
		api: "openai-completions",
		provider: "openrouter",
		baseUrl: "https://openrouter.ai/api/v1",
		reasoning: true,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 100_000,
		maxTokens: 4096,
	};
}

const copilotModel: Model<"openai-completions"> = {
	...model(),
	id: "gemini-4-neon",
	provider: "github-copilot",
	baseUrl: "https://api.individual.githubcopilot.com",
};

function chunk(delta: Record<string, unknown>, finishReason: string | null = null): unknown {
	return {
		id: "chatcmpl-test",
		model: "google/gemini-test",
		choices: [{ index: 0, delta, finish_reason: finishReason }],
	};
}

function toolCallChunk(): unknown {
	return chunk({
		tool_calls: [
			{
				index: 0,
				id: "call_1",
				type: "function",
				function: { name: "read", arguments: '{"path":"README.md"}' },
			},
		],
	});
}

async function runOpenAICompletionsStream(
	messages: AssistantMessage[] = [],
	streamModel = model(),
): Promise<AssistantMessage> {
	return await streamOpenAICompletions(streamModel, normalizeContext({ messages, tools: [readTool] }), {
		apiKey: "test",
	}).result();
}

function getAssistantPayload(payload: unknown):
	| {
			role?: string;
			reasoning?: unknown;
			reasoning_details?: unknown;
			reasoning_opaque?: unknown;
			reasoning_text?: unknown;
	  }
	| undefined {
	const messages = (
		payload as {
			messages?: Array<{
				role?: string;
				reasoning?: unknown;
				reasoning_details?: unknown;
				reasoning_opaque?: unknown;
				reasoning_text?: unknown;
			}>;
		}
	).messages;
	return messages?.find((message) => message.role === "assistant");
}

describe("openai-completions reasoning_details streaming", () => {
	beforeEach(() => {
		mockState.chunkSets = [];
		mockState.payloads = [];
	});

	it("preserves reasoning_details in the thinking signature", async () => {
		mockState.chunkSets = [
			[chunk({ reasoning_details: [reasoningDetail] }), toolCallChunk(), chunk({}, "tool_calls")],
			[chunk({ content: "ok" }), chunk({}, "stop")],
		];

		const assistantMessage = await runOpenAICompletionsStream();
		const thinking = assistantMessage.content.find((block) => block.type === "thinking");
		expect(thinking).toEqual({
			type: "thinking",
			thinking: "",
			thinkingSignature: JSON.stringify([reasoningDetail]),
		});
		const toolCall = assistantMessage.content.find((block) => block.type === "toolCall");
		expect(toolCall).toEqual({
			type: "toolCall",
			id: "call_1",
			name: "read",
			arguments: { path: "README.md" },
		});

		await runOpenAICompletionsStream([assistantMessage]);

		expect(getAssistantPayload(mockState.payloads[1])?.reasoning_details).toEqual([reasoningDetail]);
	});

	it("preserves GitHub Copilot reasoning replay fields", async () => {
		mockState.chunkSets = [
			[
				chunk({ reasoning_content: "Displayed summary.", reasoning_text: "Inspect " }),
				chunk({ reasoning_text: "the ", reasoning_opaque: "copilot-signature" }),
				chunk({ reasoning_text: "file." }),
				toolCallChunk(),
				chunk({}, "tool_calls"),
			],
			[chunk({ content: "ok" }), chunk({}, "stop")],
		];

		const assistantMessage = await runOpenAICompletionsStream([], copilotModel);
		const thinking = assistantMessage.content.find((block) => block.type === "thinking");
		expect(thinking).toEqual({
			type: "thinking",
			thinking: "Displayed summary.the file.",
			thinkingSignature: JSON.stringify({
				version: 1,
				reasoning_opaque: "copilot-signature",
				reasoning_text: "Inspect the file.",
			}),
		});

		await runOpenAICompletionsStream([assistantMessage], copilotModel);

		expect(getAssistantPayload(mockState.payloads[1])).toMatchObject({
			reasoning_opaque: "copilot-signature",
			reasoning_text: "Inspect the file.",
		});
		expect(getAssistantPayload(mockState.payloads[1])?.reasoning_details).toBeUndefined();
	});

	it("replays an opaque signature received with the Copilot tool-call finish reason", async () => {
		mockState.chunkSets = [
			[toolCallChunk(), chunk({ reasoning_opaque: "opaque-only-signature" }, "tool_calls")],
			[chunk({ content: "ok" }), chunk({}, "stop")],
		];

		const assistantMessage = await runOpenAICompletionsStream([], copilotModel);
		expect(assistantMessage.stopReason).toBe("toolUse");
		expect(assistantMessage.content.find((block) => block.type === "thinking")).toEqual({
			type: "thinking",
			thinking: "",
			thinkingSignature: JSON.stringify({
				version: 1,
				reasoning_opaque: "opaque-only-signature",
			}),
		});

		await runOpenAICompletionsStream([assistantMessage], copilotModel);

		expect(getAssistantPayload(mockState.payloads[1])).toMatchObject({
			reasoning_opaque: "opaque-only-signature",
		});
		expect(getAssistantPayload(mockState.payloads[1])?.reasoning_text).toBeUndefined();
	});

	it("still rejects a Copilot tool-call stream without a finish reason", async () => {
		mockState.chunkSets = [[toolCallChunk(), chunk({ reasoning_opaque: "copilot-signature" })]];

		const assistantMessage = await runOpenAICompletionsStream([], copilotModel);
		expect(assistantMessage).toMatchObject({
			stopReason: "error",
			errorMessage: "Stream ended without finish_reason",
		});
	});

	it("ignores opaque reasoning from other providers", async () => {
		mockState.chunkSets = [
			[
				chunk({ reasoning_text: "Other summary.", reasoning_opaque: "ignored" }),
				toolCallChunk(),
				chunk({}, "tool_calls"),
			],
			[chunk({ content: "ok" }), chunk({}, "stop")],
		];

		const assistantMessage = await runOpenAICompletionsStream();
		expect(assistantMessage.content[0]).toMatchObject({ thinkingSignature: "reasoning_text" });

		await runOpenAICompletionsStream([assistantMessage]);
		expect(getAssistantPayload(mockState.payloads[1])?.reasoning_text).toBe("Other summary.");
		expect(getAssistantPayload(mockState.payloads[1])?.reasoning_opaque).toBeUndefined();
	});

	it("does not replay Copilot metadata stored in older non-Copilot history", async () => {
		mockState.chunkSets = [[chunk({ content: "ok" }), chunk({}, "stop")]];
		const previous: AssistantMessage = {
			role: "assistant",
			api: "openai-completions",
			provider: model().provider,
			model: model().id,
			content: [
				{ type: "thinking", thinking: "Summary", thinkingSignature: '{"version":1,"reasoning_opaque":"old"}' },
				{ type: "text", text: "Answer" },
			],
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "stop",
			timestamp: 0,
		};

		await runOpenAICompletionsStream([previous]);
		expect(getAssistantPayload(mockState.payloads[0])?.reasoning_opaque).toBeUndefined();
	});

	it("falls back to encrypted tool-call signatures for older stored assistant messages", async () => {
		mockState.chunkSets = [
			[chunk({ reasoning_details: [reasoningDetail] }), toolCallChunk(), chunk({}, "tool_calls")],
			[chunk({ content: "ok" }), chunk({}, "stop")],
		];

		const assistantMessage = await runOpenAICompletionsStream();
		assistantMessage.content = assistantMessage.content.filter((block) => block.type !== "thinking");
		const toolCall = assistantMessage.content.find((block) => block.type === "toolCall");
		if (!toolCall || toolCall.type !== "toolCall") throw new Error("Expected tool call");
		toolCall.thoughtSignature = JSON.stringify(reasoningDetail);

		await runOpenAICompletionsStream([assistantMessage]);

		expect(getAssistantPayload(mockState.payloads[1])?.reasoning_details).toEqual([reasoningDetail]);
	});

	it("preserves signed text and summary reasoning_details in their original sequence", async () => {
		mockState.chunkSets = [
			[
				chunk({ reasoning: signedReasoningTextDetail.text, reasoning_details: [signedReasoningTextDetail] }),
				chunk({ reasoning_details: [reasoningDetail, reasoningSummaryDetail] }),
				toolCallChunk(),
				chunk({}, "tool_calls"),
			],
			[chunk({ content: "ok" }), chunk({}, "stop")],
		];

		const assistantMessage = await runOpenAICompletionsStream();
		const expectedReasoningDetails = [signedReasoningTextDetail, reasoningDetail, reasoningSummaryDetail];
		const thinking = assistantMessage.content.find((block) => block.type === "thinking");
		expect(thinking).toEqual({
			type: "thinking",
			thinking: signedReasoningTextDetail.text,
			thinkingSignature: JSON.stringify(expectedReasoningDetails),
		});

		await runOpenAICompletionsStream([assistantMessage]);

		const payload = getAssistantPayload(mockState.payloads[1]);
		expect(payload?.reasoning_details).toEqual(expectedReasoningDetails);
		expect(payload?.reasoning).toBeUndefined();
	});

	it("merges consecutive text and summary reasoning_details deltas before replay", async () => {
		const textDelta = { type: "reasoning.text", text: "The", index: 0 };
		const textDeltaWithSignature = {
			type: "reasoning.text",
			text: " user wants the time.",
			signature: "sha256:text-signature",
			format: "openai-responses-v1",
			index: 0,
		};
		const summaryDelta = { type: "reasoning.summary", summary: "Looked", index: 0 };
		const summaryDeltaWithFormat = {
			type: "reasoning.summary",
			summary: " up time.",
			format: "openai-responses-v1",
			index: 0,
		};
		const laterSummaryDelta = {
			type: "reasoning.summary",
			summary: "After encrypted block.",
			format: "openai-responses-v1",
			index: 0,
		};
		const expectedReasoningDetails = [
			{
				type: "reasoning.text",
				text: "The user wants the time.",
				index: 0,
				signature: "sha256:text-signature",
				format: "openai-responses-v1",
			},
			{
				type: "reasoning.summary",
				summary: "Looked up time.",
				index: 0,
				format: "openai-responses-v1",
			},
			reasoningDetail,
			laterSummaryDelta,
		];

		mockState.chunkSets = [
			[
				chunk({ reasoning_details: [textDelta] }),
				chunk({ reasoning_details: [textDeltaWithSignature] }),
				chunk({ reasoning_details: [summaryDelta] }),
				chunk({ reasoning_details: [summaryDeltaWithFormat] }),
				chunk({ reasoning_details: [reasoningDetail] }),
				chunk({ reasoning_details: [laterSummaryDelta] }),
				toolCallChunk(),
				chunk({}, "tool_calls"),
			],
			[chunk({ content: "ok" }), chunk({}, "stop")],
		];

		const assistantMessage = await runOpenAICompletionsStream();
		const thinking = assistantMessage.content.find((block) => block.type === "thinking");
		expect(thinking).toEqual({
			type: "thinking",
			thinking: "",
			thinkingSignature: JSON.stringify(expectedReasoningDetails),
		});

		await runOpenAICompletionsStream([assistantMessage]);

		expect(getAssistantPayload(mockState.payloads[1])?.reasoning_details).toEqual(expectedReasoningDetails);
	});
});

describe("openai-completions reasoning and content in one delta", () => {
	beforeEach(() => {
		mockState.chunkSets = [];
		mockState.payloads = [];
	});

	// https://github.com/earendil-works/pi/issues/10926
	it("places thinking before text when one delta carries reasoning_content and content", async () => {
		mockState.chunkSets = [[chunk({ role: "assistant", reasoning_content: "thinking", content: "answer" }, "stop")]];

		const assistantMessage = await runOpenAICompletionsStream();
		expect(assistantMessage.content).toEqual([
			{ type: "thinking", thinking: "thinking", thinkingSignature: "reasoning_content" },
			{ type: "text", text: "answer" },
		]);
	});

	it("places thinking before text when one delta carries reasoning_details and content", async () => {
		mockState.chunkSets = [[chunk({ reasoning_details: [reasoningDetail], content: "answer" }, "stop")]];

		const assistantMessage = await runOpenAICompletionsStream();
		expect(assistantMessage.content.map((block) => block.type)).toEqual(["thinking", "text"]);
	});
});
