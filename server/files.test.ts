import { describe, expect, it } from "vitest";
import { appRouter } from "./routers";
import type { TrpcContext } from "./_core/context";

function unauthenticatedContext(): TrpcContext {
  return {
    user: null,
    req: { protocol: "https", headers: {} } as TrpcContext["req"],
    res: {} as TrpcContext["res"],
  };
}

describe("files router", () => {
  it("protects the stored file list", async () => {
    const caller = appRouter.createCaller(unauthenticatedContext());
    await expect(caller.files.list()).rejects.toMatchObject({ code: "UNAUTHORIZED" });
  });

  it("protects file retrieval", async () => {
    const caller = appRouter.createCaller(unauthenticatedContext());
    await expect(caller.files.get({ id: 1 })).rejects.toMatchObject({ code: "UNAUTHORIZED" });
  });

  it("protects file removal", async () => {
    const caller = appRouter.createCaller(unauthenticatedContext());
    await expect(caller.files.remove({ id: 1 })).rejects.toMatchObject({ code: "UNAUTHORIZED" });
  });
});
