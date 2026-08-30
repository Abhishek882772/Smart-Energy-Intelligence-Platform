import { COOKIE_NAME } from "@shared/const";
import { z } from "zod";
import { getSessionCookieOptions } from "./_core/cookies";
import { systemRouter } from "./_core/systemRouter";
import { protectedProcedure, publicProcedure, router } from "./_core/trpc";
import { deleteStoredFileByIdAndUser, getStoredFileByIdAndUser, insertStoredFile, listStoredFilesByUser } from "./db";
import { storagePut } from "./storage";
import { TRPCError } from "@trpc/server";

const MAX_FILE_BYTES = 12 * 1024 * 1024;
const base64FileSchema = z.object({
  fileName: z.string().trim().min(1).max(255),
  mimeType: z.string().trim().min(1).max(128),
  sizeBytes: z.number().int().positive().max(MAX_FILE_BYTES),
  contentBase64: z.string().min(1).max(Math.ceil(MAX_FILE_BYTES * 1.4)),
});

function safeFileName(fileName: string) {
  return fileName.replace(/[^a-zA-Z0-9._ -]/g, "_").slice(0, 180) || "upload.bin";
}

export const appRouter = router({
  system: systemRouter,
  auth: router({
    me: publicProcedure.query(opts => opts.ctx.user),
    logout: publicProcedure.mutation(({ ctx }) => {
      const cookieOptions = getSessionCookieOptions(ctx.req);
      ctx.res.clearCookie(COOKIE_NAME, { ...cookieOptions, maxAge: -1 });
      return { success: true } as const;
    }),
  }),
  files: router({
    list: protectedProcedure.query(({ ctx }) => listStoredFilesByUser(ctx.user.id)),
    get: protectedProcedure.input(z.object({ id: z.number().int().positive() })).query(async ({ ctx, input }) => {
      const file = await getStoredFileByIdAndUser(input.id, ctx.user.id);
      if (!file) throw new TRPCError({ code: "NOT_FOUND", message: "File not found." });
      return file;
    }),
    upload: protectedProcedure.input(base64FileSchema).mutation(async ({ ctx, input }) => {
      const fileBuffer = Buffer.from(input.contentBase64, "base64");
      if (fileBuffer.byteLength !== input.sizeBytes) {
        throw new TRPCError({ code: "BAD_REQUEST", message: "The uploaded file size does not match its metadata." });
      }
      const normalizedName = safeFileName(input.fileName);
      const relativeKey = `${ctx.user.id}/reports/${Date.now()}-${normalizedName}`;
      const stored = await storagePut(relativeKey, fileBuffer, input.mimeType);
      return insertStoredFile({
        userId: ctx.user.id,
        fileKey: stored.key,
        url: stored.url,
        fileName: input.fileName,
        mimeType: input.mimeType,
        sizeBytes: input.sizeBytes,
      });
    }),
    remove: protectedProcedure.input(z.object({ id: z.number().int().positive() })).mutation(async ({ ctx, input }) => {
      const deleted = await deleteStoredFileByIdAndUser(input.id, ctx.user.id);
      if (!deleted) throw new TRPCError({ code: "NOT_FOUND", message: "File not found." });
      return { success: true as const };
    }),
  }),
});

export type AppRouter = typeof appRouter;
