import type { LedgerMem } from "@ledgermem/memory";

/** Structural subset of `Anthropic` we depend on — keeps the peer dep loose. */
interface AnthropicLike {
  messages: {
    create: (params: Record<string, unknown>) => Promise<MessagesCreateResult>;
  };
}

interface MessagesCreateResult {
  id: string;
  content: Array<MessageBlock>;
  stop_reason: string;
  usage?: Record<string, number>;
  [key: string]: unknown;
}

type MessageBlock =
  | { type: "text"; text: string }
  | { type: "thinking"; thinking: string }
  | {
      type: "tool_use";
      id: string;
      name: string;
      input: Record<string, unknown>;
    };

interface ChatMessage {
  role: "user" | "assistant";
  content: string | Array<Record<string, unknown>>;
}

export interface WithMemoryToolOptions {
  client: AnthropicLike;
  ledgermem: LedgerMem;
  /** Defaults to claude-sonnet-4-7. Override per call too. */
  model?: string;
  /** Defaults to 1024. */
  maxTokens?: number;
  /** Default top-k for memory search. */
  searchLimit?: number;
  /** Wrap the system prompt with `cache_control: { type: "ephemeral" }`. */
  cacheSystem?: boolean;
  /** Enable extended thinking with this token budget. */
  thinkingBudgetTokens?: number;
  /** Extra static metadata merged into every persisted memory. */
  metadata?: Record<string, unknown>;
  /** Hard cap on tool-use loop iterations. */
  maxIterations?: number;
}

export interface WithMemoryRunInput {
  system?: string;
  messages: ChatMessage[];
  /** Per-call overrides. */
  model?: string;
  maxTokens?: number;
  metadata?: Record<string, unknown>;
}

export interface WithMemoryRunResult {
  /** Final assistant text concatenated across blocks. */
  text: string;
  /** All raw API responses produced during the tool-use loop. */
  responses: MessagesCreateResult[];
  /** Number of memory tool invocations. */
  memoryToolCalls: number;
  /** Final messages array after tool loop (useful for storing transcript). */
  messages: ChatMessage[];
}

export interface WithMemoryToolWrapper {
  run(input: WithMemoryRunInput): Promise<WithMemoryRunResult>;
}

export const MEMORY_TOOL_NAME = "memory";

/**
 * Wrap an `Anthropic` client so a single `memory` tool (with `search`/`add`
 * actions) is exposed to the model and the tool-use loop is handled for you.
 *
 * Also opts into prompt caching of the system prompt and (optionally) extended
 * thinking — both are recommended for any long-lived assistant.
 */
export function withMemoryTool(
  options: WithMemoryToolOptions,
): WithMemoryToolWrapper {
  const defaults = {
    model: options.model ?? "claude-sonnet-4-7",
    maxTokens: options.maxTokens ?? 1024,
    searchLimit: options.searchLimit ?? 5,
    cacheSystem: options.cacheSystem ?? true,
    thinkingBudgetTokens: options.thinkingBudgetTokens,
    metadata: options.metadata ?? {},
    maxIterations: options.maxIterations ?? 6,
  };

  const tool = {
    name: MEMORY_TOOL_NAME,
    description:
      "Long-term memory store. Use action='search' to recall facts about the user before answering, and action='add' to save a new fact worth remembering across conversations.",
    input_schema: {
      type: "object" as const,
      properties: {
        action: {
          type: "string",
          enum: ["search", "add"],
          description: "Whether to read from or write to memory.",
        },
        query: {
          type: "string",
          description: "Required when action='search'. Natural-language query.",
        },
        content: {
          type: "string",
          description: "Required when action='add'. The fact to remember.",
        },
        limit: {
          type: "integer",
          minimum: 1,
          maximum: 50,
          description: "Optional max results for search.",
        },
        tags: {
          type: "object",
          description: "Optional metadata merged into the stored memory.",
          additionalProperties: true,
        },
      },
      required: ["action"],
    },
  };

  return {
    async run(input: WithMemoryRunInput): Promise<WithMemoryRunResult> {
      const messages: ChatMessage[] = [...input.messages];
      const responses: MessagesCreateResult[] = [];
      let toolCalls = 0;

      const system = buildSystem(input.system, defaults.cacheSystem);
      const baseParams: Record<string, unknown> = {
        model: input.model ?? defaults.model,
        max_tokens: input.maxTokens ?? defaults.maxTokens,
        tools: [tool],
        ...(system ? { system } : {}),
      };
      if (defaults.thinkingBudgetTokens) {
        baseParams.thinking = {
          type: "enabled",
          budget_tokens: defaults.thinkingBudgetTokens,
        };
      }

      for (let i = 0; i < defaults.maxIterations; i++) {
        const resp = await options.client.messages.create({
          ...baseParams,
          messages,
        });
        responses.push(resp);

        const toolUses = resp.content.filter(
          (b): b is Extract<MessageBlock, { type: "tool_use" }> =>
            b.type === "tool_use" && b.name === MEMORY_TOOL_NAME,
        );

        messages.push({ role: "assistant", content: resp.content });

        if (resp.stop_reason !== "tool_use" || toolUses.length === 0) {
          return finalize(messages, responses, toolCalls);
        }

        const toolResults: Array<Record<string, unknown>> = [];
        for (const use of toolUses) {
          toolCalls += 1;
          const result = await runMemoryTool(
            use.input,
            options.ledgermem,
            { ...defaults.metadata, ...(input.metadata ?? {}) },
            defaults.searchLimit,
          );
          toolResults.push({
            type: "tool_result",
            tool_use_id: use.id,
            content: JSON.stringify(result),
          });
        }
        messages.push({ role: "user", content: toolResults });
      }

      return finalize(messages, responses, toolCalls);
    },
  };
}

function finalize(
  messages: ChatMessage[],
  responses: MessagesCreateResult[],
  memoryToolCalls: number,
): WithMemoryRunResult {
  const last = responses[responses.length - 1];
  const text = last
    ? last.content
        .filter((b): b is Extract<MessageBlock, { type: "text" }> => b.type === "text")
        .map((b) => b.text)
        .join("\n")
        .trim()
    : "";
  return { text, responses, memoryToolCalls, messages };
}

function buildSystem(
  system: string | undefined,
  cache: boolean,
): Array<Record<string, unknown>> | undefined {
  if (!system) return undefined;
  if (!cache) return [{ type: "text", text: system }];
  return [
    { type: "text", text: system, cache_control: { type: "ephemeral" } },
  ];
}

async function runMemoryTool(
  raw: Record<string, unknown>,
  client: LedgerMem,
  baseMetadata: Record<string, unknown>,
  defaultLimit: number,
): Promise<unknown> {
  const action = String(raw.action ?? "");
  if (action === "search") {
    const query = String(raw.query ?? "").trim();
    if (!query) return { error: "query is required for search" };
    const limit = Number(raw.limit ?? defaultLimit) || defaultLimit;
    const results = await client.search(query, { limit });
    return { results };
  }
  if (action === "add") {
    const content = String(raw.content ?? "").trim();
    if (!content) return { error: "content is required for add" };
    const tags =
      typeof raw.tags === "object" && raw.tags
        ? (raw.tags as Record<string, unknown>)
        : {};
    const memory = await client.add(content, {
      metadata: { ...baseMetadata, ...tags },
    });
    return { memory };
  }
  return { error: `unknown action: ${action}` };
}
