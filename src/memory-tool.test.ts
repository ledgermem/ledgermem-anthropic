import { describe, expect, it, vi } from "vitest";
import { withMemoryTool, MEMORY_TOOL_NAME } from "./memory-tool.js";

function fakeLedgerMem() {
  return {
    search: vi
      .fn()
      .mockResolvedValue([{ id: "m1", content: "user prefers dark mode" }]),
    add: vi.fn().mockResolvedValue({ id: "m2", content: "stored" }),
    update: vi.fn(),
    delete: vi.fn(),
    list: vi.fn(),
  } as any;
}

describe("withMemoryTool", () => {
  it("short-circuits when the model returns text without tool use", async () => {
    const client = {
      messages: {
        create: vi.fn().mockResolvedValue({
          id: "r1",
          stop_reason: "end_turn",
          content: [{ type: "text", text: "Hi there." }],
        }),
      },
    };
    const wrap = withMemoryTool({ client, ledgermem: fakeLedgerMem() });
    const out = await wrap.run({
      system: "You are helpful.",
      messages: [{ role: "user", content: "Hello" }],
    });
    expect(out.text).toBe("Hi there.");
    expect(out.memoryToolCalls).toBe(0);
    expect(client.messages.create).toHaveBeenCalledOnce();
  });

  it("runs the memory tool loop and feeds tool_result back", async () => {
    const ledgermem = fakeLedgerMem();
    const client = {
      messages: {
        create: vi
          .fn()
          .mockResolvedValueOnce({
            id: "r1",
            stop_reason: "tool_use",
            content: [
              {
                type: "tool_use",
                id: "tu_1",
                name: MEMORY_TOOL_NAME,
                input: { action: "search", query: "ui prefs" },
              },
            ],
          })
          .mockResolvedValueOnce({
            id: "r2",
            stop_reason: "end_turn",
            content: [{ type: "text", text: "You prefer dark mode." }],
          }),
      },
    };
    const wrap = withMemoryTool({ client, ledgermem });
    const out = await wrap.run({
      messages: [{ role: "user", content: "What do I prefer?" }],
    });
    expect(out.text).toBe("You prefer dark mode.");
    expect(out.memoryToolCalls).toBe(1);
    expect(ledgermem.search).toHaveBeenCalledWith("ui prefs", { limit: 5 });
    // 2nd call should include the tool_result
    const secondCall = client.messages.create.mock.calls[1]![0] as any;
    const toolResultMsg = secondCall.messages.at(-1);
    expect(toolResultMsg.role).toBe("user");
    expect((toolResultMsg.content as any[])[0].type).toBe("tool_result");
  });

  it("opts into prompt caching on the system block by default", async () => {
    const client = {
      messages: {
        create: vi.fn().mockResolvedValue({
          id: "r1",
          stop_reason: "end_turn",
          content: [{ type: "text", text: "ok" }],
        }),
      },
    };
    const wrap = withMemoryTool({ client, ledgermem: fakeLedgerMem() });
    await wrap.run({
      system: "stable instructions",
      messages: [{ role: "user", content: "hi" }],
    });
    const params = client.messages.create.mock.calls[0]![0] as any;
    expect(params.system[0].cache_control).toEqual({ type: "ephemeral" });
  });

  it("enables extended thinking when configured", async () => {
    const client = {
      messages: {
        create: vi.fn().mockResolvedValue({
          id: "r1",
          stop_reason: "end_turn",
          content: [{ type: "text", text: "ok" }],
        }),
      },
    };
    const wrap = withMemoryTool({
      client,
      ledgermem: fakeLedgerMem(),
      thinkingBudgetTokens: 4000,
    });
    await wrap.run({ messages: [{ role: "user", content: "hi" }] });
    const params = client.messages.create.mock.calls[0]![0] as any;
    expect(params.thinking).toEqual({ type: "enabled", budget_tokens: 4000 });
  });

  it("respects maxIterations and stops the loop", async () => {
    const ledgermem = fakeLedgerMem();
    const client = {
      messages: {
        create: vi.fn().mockResolvedValue({
          id: "r",
          stop_reason: "tool_use",
          content: [
            {
              type: "tool_use",
              id: "tu",
              name: MEMORY_TOOL_NAME,
              input: { action: "search", query: "x" },
            },
          ],
        }),
      },
    };
    const wrap = withMemoryTool({
      client,
      ledgermem,
      maxIterations: 2,
    });
    const out = await wrap.run({
      messages: [{ role: "user", content: "loop" }],
    });
    expect(client.messages.create).toHaveBeenCalledTimes(2);
    expect(out.memoryToolCalls).toBe(2);
  });
});
