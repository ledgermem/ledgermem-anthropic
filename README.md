# @ledgermem/anthropic

LedgerMem helper for the Anthropic SDK. Wraps `messages.create` with a single
`memory` tool (search + add), prompt caching on the system block, and
optional extended thinking — the recommended setup for any long-lived Claude
assistant.

## Install

```bash
npm install @ledgermem/anthropic @ledgermem/memory @anthropic-ai/sdk
```

## Quickstart (30 seconds)

```ts
import Anthropic from "@anthropic-ai/sdk";
import { LedgerMem } from "@ledgermem/memory";
import { withMemoryTool } from "@ledgermem/anthropic";

const claude = new Anthropic();
const ledgermem = new LedgerMem({
  apiKey: process.env.LEDGERMEM_API_KEY!,
  workspaceId: process.env.LEDGERMEM_WORKSPACE_ID!,
});

const agent = withMemoryTool({
  client: claude,
  ledgermem,
  model: "claude-sonnet-4-7",
  thinkingBudgetTokens: 4000,
});

const result = await agent.run({
  system: "You are a personal assistant. Use memory before answering.",
  messages: [{ role: "user", content: "What did I tell you about my schedule?" }],
  metadata: { userId: "user-42" },
});

console.log(result.text);
console.log("memory tool calls:", result.memoryToolCalls);
```

## What the wrapper does

- Exposes one tool `memory` with `action: "search" | "add"` — the model
  decides when to recall and when to remember.
- Runs the tool-use loop for you (default cap: 6 iterations).
- Adds `cache_control: { type: "ephemeral" }` to the system prompt so
  long stable instructions hit the prompt cache.
- Optionally enables extended thinking with a configurable token budget.
- Returns the final transcript so you can persist it elsewhere.

## License

MIT
