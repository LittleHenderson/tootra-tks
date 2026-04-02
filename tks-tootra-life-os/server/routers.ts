import { COOKIE_NAME } from "@shared/const";
import { getSessionCookieOptions } from "./_core/cookies";
import { systemRouter } from "./_core/systemRouter";
import { publicProcedure, protectedProcedure, router } from "./_core/trpc";
import { z } from "zod";
import * as db from "./db";
import { interpretTask, generateSubtasksForGap, buildScenarioPrompts } from "./ai";
import { generateImage } from "./_core/imageGeneration";
import { performInversion } from "./inversion";
import { INVERSION_MODES, INVERSION_PRESETS, INVERSION_LAYERS, getInversionMode, getPreset, getPresetModes, buildDefaultDialConfig } from "../shared/inversions";
import { calculatePriorityScore } from "./priority";
import { transcribeAudio } from "./_core/voiceTranscription";
import { storagePut } from "./storage";
import { TRPCError } from "@trpc/server";
import { nanoid } from "nanoid";

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

  // ── User Settings ───────────────────────────────────────
  settings: router({
    getMode: protectedProcedure.query(async ({ ctx }) => {
      return { mode: ctx.user.mode || "life_os" };
    }),
    setMode: protectedProcedure
      .input(z.object({ mode: z.enum(["life_os", "work_sprint"]) }))
      .mutation(async ({ ctx, input }) => {
        await db.updateUserMode(ctx.user.id, input.mode);
        return { mode: input.mode };
      }),
  }),

  // ── Tasks ───────────────────────────────────────────────
  tasks: router({
    list: protectedProcedure
      .input(z.object({ status: z.string().optional() }).optional())
      .query(async ({ ctx, input }) => {
        return db.getTasksByUser(ctx.user.id, input?.status);
      }),

    get: protectedProcedure
      .input(z.object({ id: z.number() }))
      .query(async ({ ctx, input }) => {
        return db.getTaskById(input.id, ctx.user.id);
      }),

    create: protectedProcedure
      .input(z.object({ rawInput: z.string().min(1) }))
      .mutation(async ({ ctx, input }) => {
        // AI interprets the task
        const interpretation = await interpretTask(input.rawInput);

        // Create the task with all TKS data
        const task = await db.createTask({
          userId: ctx.user.id,
          title: interpretation.title,
          rawInput: input.rawInput,
          description: interpretation.description,
          status: "inbox",
          archetype: interpretation.archetype as any,
          foundationId: interpretation.foundationId,
          subFoundationId: interpretation.subFoundationId,
          mappingConfidence: interpretation.mappingConfidence,
          desireStage: interpretation.desireStage,
          wisdomStage: interpretation.wisdomStage,
          powerStage: interpretation.powerStage,
          equationCanonical: interpretation.equationCanonical,
          equationAst: interpretation.equationAst,
          urgency: interpretation.urgency as any,
          durationMinutes: interpretation.durationMinutes,
          dwpDiagnosis: interpretation.dwpDiagnosis,
          priorityScore: interpretation.priorityScore,
        });

        // Save power inventory
        if (task && interpretation.powerInventory) {
          for (const item of interpretation.powerInventory) {
            await db.upsertPowerItem({
              taskId: task.id,
              userId: ctx.user.id,
              category: item.category as any,
              status: item.status as any,
              notes: item.note,
              suggestedSubtask: item.suggestedSubtask || null,
            });
          }
        }

        return { task, interpretation };
      }),

    update: protectedProcedure
      .input(z.object({
        id: z.number(),
        title: z.string().optional(),
        description: z.string().optional(),
        status: z.enum(["inbox", "scheduled", "active", "done", "dropped"]).optional(),
        archetype: z.enum(["execution", "prep", "maintenance", "repair", "study", "audit"]).optional(),
        desireStage: z.string().optional(),
        wisdomStage: z.string().optional(),
        powerStage: z.string().optional(),
        urgency: z.enum(["low", "medium", "high", "critical"]).optional(),
        scheduledAt: z.string().optional(),
        dueAt: z.string().optional(),
        durationMinutes: z.number().optional(),
      }))
      .mutation(async ({ ctx, input }) => {
        const { id, scheduledAt, dueAt, ...rest } = input;
        const updateData: any = { ...rest };
        if (scheduledAt) updateData.scheduledAt = new Date(scheduledAt);
        if (dueAt) updateData.dueAt = new Date(dueAt);
        if (input.status === "done") updateData.completedAt = new Date();
        return db.updateTask(id, ctx.user.id, updateData);
      }),

    delete: protectedProcedure
      .input(z.object({ id: z.number() }))
      .mutation(async ({ ctx, input }) => {
        await db.deleteTask(input.id, ctx.user.id);
        return { success: true };
      }),

    recalculatePriority: protectedProcedure
      .input(z.object({ id: z.number() }))
      .mutation(async ({ ctx, input }) => {
        const task = await db.getTaskById(input.id, ctx.user.id);
        if (!task) throw new Error("Task not found");
        const allTasks = await db.getTasksByUser(ctx.user.id);
        const foundationCounts: Record<number, number> = {};
        for (const t of allTasks) {
          const fId = t.foundationId || 0;
          foundationCounts[fId] = (foundationCounts[fId] || 0) + 1;
        }
        const powerItems = await db.getPowerInventory(task.id, ctx.user.id);
        const score = calculatePriorityScore({
          dueAt: task.dueAt,
          foundationId: task.foundationId,
          foundationCounts,
          desireStage: task.desireStage,
          powerItems: powerItems.map(p => ({ status: p.status })),
          urgency: task.urgency,
          recurrence: task.recurrence,
        });
        await db.updateTask(task.id, ctx.user.id, { priorityScore: score });
        return { score };
      }),
  }),

  // ── Power Inventory ─────────────────────────────────────
  power: router({
    getForTask: protectedProcedure
      .input(z.object({ taskId: z.number() }))
      .query(async ({ ctx, input }) => {
        return db.getPowerInventory(input.taskId, ctx.user.id);
      }),

    update: protectedProcedure
      .input(z.object({
        taskId: z.number(),
        category: z.enum(["time", "money", "tools", "access", "body", "skill", "connections", "environment"]),
        status: z.enum(["have", "lack", "partial"]),
        notes: z.string().optional(),
      }))
      .mutation(async ({ ctx, input }) => {
        return db.upsertPowerItem({
          taskId: input.taskId,
          userId: ctx.user.id,
          category: input.category,
          status: input.status,
          notes: input.notes || null,
          suggestedSubtask: null,
        });
      }),

    generateSubtasks: protectedProcedure
      .input(z.object({ taskTitle: z.string(), category: z.string() }))
      .mutation(async ({ input }) => {
        const subtasks = await generateSubtasksForGap(input.taskTitle, input.category);
        return { subtasks };
      }),
  }),

  // ── Voice Transcription ─────────────────────────────────
  voice: router({
    transcribe: protectedProcedure
      .input(z.object({
        audioBase64: z.string().min(1),
        mimeType: z.string().default("audio/webm"),
        language: z.string().optional(),
      }))
      .mutation(async ({ ctx, input }) => {
        // Decode base64 audio
        const audioBuffer = Buffer.from(input.audioBase64, "base64");

        // Check size (16MB limit)
        const sizeMB = audioBuffer.length / (1024 * 1024);
        if (sizeMB > 16) {
          throw new TRPCError({
            code: "BAD_REQUEST",
            message: `Audio file is ${sizeMB.toFixed(1)}MB — maximum is 16MB. Try a shorter recording.`,
          });
        }

        // Upload to S3
        const ext = input.mimeType.includes("webm") ? "webm" : input.mimeType.includes("mp4") ? "m4a" : "wav";
        const fileKey = `voice/${ctx.user.id}/${nanoid()}.${ext}`;
        const { url: audioUrl } = await storagePut(fileKey, audioBuffer, input.mimeType);

        // Transcribe
        const result = await transcribeAudio({
          audioUrl,
          language: input.language || "en",
          prompt: "Transcribe this task description spoken by the user.",
        });

        if ("error" in result) {
          throw new TRPCError({
            code: "INTERNAL_SERVER_ERROR",
            message: result.error,
            cause: result,
          });
        }

        return {
          text: result.text,
          language: result.language,
          duration: result.duration,
        };
      }),
  }),

  // ── Scenarioize (Vision) ────────────────────────────────
  scenario: router({
    uploadPhoto: protectedProcedure
      .input(z.object({
        photoBase64: z.string().min(1),
        mimeType: z.string().default("image/jpeg"),
      }))
      .mutation(async ({ ctx, input }) => {
        const buffer = Buffer.from(input.photoBase64, "base64");
        const sizeMB = buffer.length / (1024 * 1024);
        if (sizeMB > 10) {
          throw new TRPCError({ code: "BAD_REQUEST", message: `Photo is ${sizeMB.toFixed(1)}MB — maximum is 10MB.` });
        }
        const ext = input.mimeType.includes("png") ? "png" : "jpg";
        const fileKey = `photos/${ctx.user.id}/${nanoid()}.${ext}`;
        const { url } = await storagePut(fileKey, buffer, input.mimeType);
        await db.updateUserPhoto(ctx.user.id, url);
        return { photoUrl: url };
      }),

    getPhoto: protectedProcedure.query(async ({ ctx }) => {
      const photoUrl = await db.getUserPhoto(ctx.user.id);
      return { photoUrl };
    }),

    generate: protectedProcedure
      .input(z.object({
        taskId: z.number(),
        mode: z.enum(["single", "dual"]).default("single"),
        type: z.enum(["positive", "negative"]).default("positive"),
      }))
      .mutation(async ({ ctx, input }) => {
        const task = await db.getTaskById(input.taskId, ctx.user.id);
        if (!task) throw new TRPCError({ code: "NOT_FOUND", message: "Task not found" });

        // Get user photo for reference
        const userPhoto = await db.getUserPhoto(ctx.user.id);

        // Build scenario prompts using AI
        const prompts = await buildScenarioPrompts({
          title: task.title,
          description: task.description,
          foundationId: task.foundationId,
          archetype: task.archetype,
          desireStage: task.desireStage,
          wisdomStage: task.wisdomStage,
          powerStage: task.powerStage,
          dwpDiagnosis: task.dwpDiagnosis,
        });

        const results: Array<{ type: "positive" | "negative"; imageUrl: string; description: string }> = [];

        // Generate based on mode
        const typesToGenerate = input.mode === "dual"
          ? ["positive", "negative"] as const
          : [input.type] as const;

        for (const scType of typesToGenerate) {
          const prompt = scType === "positive" ? prompts.positivePrompt : prompts.negativePrompt;
          const desc = scType === "positive" ? prompts.positiveDescription : prompts.negativeDescription;

          // Generate image, optionally with user photo as reference
          const genOptions: any = { prompt };
          if (userPhoto) {
            genOptions.originalImages = [{ url: userPhoto, mimeType: "image/jpeg" }];
          }

          const { url: imageUrl } = await generateImage(genOptions);

          if (imageUrl) {
            // Save to DB
            const scenario = await db.createScenario({
              taskId: input.taskId,
              userId: ctx.user.id,
              type: scType,
              prompt,
              imageUrl,
              description: desc,
            });
            results.push({ type: scType, imageUrl, description: desc });
          }
        }

        return { scenarios: results };
      }),

    listForTask: protectedProcedure
      .input(z.object({ taskId: z.number() }))
      .query(async ({ ctx, input }) => {
        return db.getScenariosForTask(input.taskId, ctx.user.id);
      }),
  }),

  // ── Inversions (TKS Inversion Engine) ───────────────────
  inversion: router({
    // Get all 29 modes and presets (static data)
    getModes: publicProcedure.query(() => {
      return {
        modes: INVERSION_MODES,
        presets: INVERSION_PRESETS,
        layers: INVERSION_LAYERS,
      };
    }),

    // Perform an inversion on a task
    perform: protectedProcedure
      .input(z.object({
        taskId: z.number(),
        modeIds: z.array(z.number()).min(1).max(5),
        intensity: z.enum(["soft", "medium", "hard"]).default("medium"),
        scope: z.enum(["local", "term", "chain", "equation", "scenario"]).default("equation"),
        direction: z.enum(["forward", "backward", "bidirectional"]).default("forward"),
        presetId: z.string().optional(),
        generateImage: z.boolean().default(false),
      }))
      .mutation(async ({ ctx, input }) => {
        const task = await db.getTaskById(input.taskId, ctx.user.id);
        if (!task) throw new TRPCError({ code: "NOT_FOUND", message: "Task not found" });

        // Build dial config
        const dialConfig = buildDefaultDialConfig(input.modeIds);
        dialConfig.intensity = input.intensity;
        dialConfig.scope = input.scope;
        dialConfig.direction = input.direction;

        // Perform the inversion via LLM
        const result = await performInversion({
          taskTitle: task.title,
          taskDescription: task.description,
          equationCanonical: task.equationCanonical || "",
          foundationId: task.foundationId,
          subFoundationId: task.subFoundationId,
          archetype: task.archetype,
          desireStage: task.desireStage,
          wisdomStage: task.wisdomStage,
          powerStage: task.powerStage,
          dwpDiagnosis: task.dwpDiagnosis,
          dialConfig,
        });

        // Optionally generate an image for the inverted scenario
        let imageUrl: string | null = null;
        if (input.generateImage && result.scenarioPrompt) {
          try {
            const userPhoto = await db.getUserPhoto(ctx.user.id);
            const genOptions: any = { prompt: result.scenarioPrompt };
            if (userPhoto) {
              genOptions.originalImages = [{ url: userPhoto, mimeType: "image/jpeg" }];
            }
            const { url } = await generateImage(genOptions);
            imageUrl = url || null;
          } catch (e) {
            console.warn("[Inversion] Image generation failed:", e);
          }
        }

        // Save to database
        const inversion = await db.createInversion({
          taskId: input.taskId,
          userId: ctx.user.id,
          modeIds: input.modeIds,
          intensity: input.intensity,
          scope: input.scope,
          presetId: input.presetId || null,
          originalEquation: result.originalEquation,
          originalNarrative: result.originalNarrative,
          invertedEquation: result.invertedEquation,
          invertedNarrative: result.invertedNarrative,
          changeSummary: result.changeSummary,
          invertedFoundationId: result.invertedFoundationId,
          invertedArchetype: result.invertedArchetype,
          invertedDesireStage: result.invertedDesireStage,
          invertedWisdomStage: result.invertedWisdomStage,
          invertedPowerStage: result.invertedPowerStage,
          scenarioPrompt: result.scenarioPrompt,
          imageUrl,
        });

        return {
          inversion,
          result,
          imageUrl,
        };
      }),

    // Apply a preset recipe
    applyPreset: protectedProcedure
      .input(z.object({
        taskId: z.number(),
        presetId: z.string(),
        generateImage: z.boolean().default(false),
      }))
      .mutation(async ({ ctx, input }) => {
        const preset = getPreset(input.presetId);
        if (!preset) throw new TRPCError({ code: "NOT_FOUND", message: "Preset not found" });

        const task = await db.getTaskById(input.taskId, ctx.user.id);
        if (!task) throw new TRPCError({ code: "NOT_FOUND", message: "Task not found" });

        const dialConfig = buildDefaultDialConfig(preset.modeIds);
        dialConfig.intensity = preset.intensity;
        dialConfig.scope = preset.scope;

        const result = await performInversion({
          taskTitle: task.title,
          taskDescription: task.description,
          equationCanonical: task.equationCanonical || "",
          foundationId: task.foundationId,
          subFoundationId: task.subFoundationId,
          archetype: task.archetype,
          desireStage: task.desireStage,
          wisdomStage: task.wisdomStage,
          powerStage: task.powerStage,
          dwpDiagnosis: task.dwpDiagnosis,
          dialConfig,
        });

        let imageUrl: string | null = null;
        if (input.generateImage && result.scenarioPrompt) {
          try {
            const userPhoto = await db.getUserPhoto(ctx.user.id);
            const genOptions: any = { prompt: result.scenarioPrompt };
            if (userPhoto) {
              genOptions.originalImages = [{ url: userPhoto, mimeType: "image/jpeg" }];
            }
            const { url } = await generateImage(genOptions);
            imageUrl = url || null;
          } catch (e) {
            console.warn("[Inversion] Image generation failed:", e);
          }
        }

        const inversion = await db.createInversion({
          taskId: input.taskId,
          userId: ctx.user.id,
          modeIds: preset.modeIds,
          intensity: preset.intensity,
          scope: preset.scope,
          presetId: input.presetId,
          originalEquation: result.originalEquation,
          originalNarrative: result.originalNarrative,
          invertedEquation: result.invertedEquation,
          invertedNarrative: result.invertedNarrative,
          changeSummary: result.changeSummary,
          invertedFoundationId: result.invertedFoundationId,
          invertedArchetype: result.invertedArchetype,
          invertedDesireStage: result.invertedDesireStage,
          invertedWisdomStage: result.invertedWisdomStage,
          invertedPowerStage: result.invertedPowerStage,
          scenarioPrompt: result.scenarioPrompt,
          imageUrl,
        });

        return { inversion, result, imageUrl, preset };
      }),

    // List inversions for a task
    listForTask: protectedProcedure
      .input(z.object({ taskId: z.number() }))
      .query(async ({ ctx, input }) => {
        return db.getInversionsForTask(input.taskId, ctx.user.id);
      }),

    // Get a single inversion by ID
    getById: protectedProcedure
      .input(z.object({ id: z.number() }))
      .query(async ({ ctx, input }) => {
        const inv = await db.getInversionById(input.id, ctx.user.id);
        if (!inv) throw new TRPCError({ code: "NOT_FOUND", message: "Inversion not found" });
        return inv;
      }),
  }),

  // ── Analytics ───────────────────────────────────────────
  analytics: router({
    heatmap: protectedProcedure.query(async ({ ctx }) => {
      return db.getFoundationHeatmap(ctx.user.id);
    }),

    weeklyStats: protectedProcedure
      .input(z.object({
        weekStart: z.string(),
        weekEnd: z.string(),
      }))
      .query(async ({ ctx, input }) => {
        return db.getWeeklyStats(ctx.user.id, new Date(input.weekStart), new Date(input.weekEnd));
      }),

    reviewHistory: protectedProcedure.query(async ({ ctx }) => {
      return db.getReviewSnapshots(ctx.user.id);
    }),

    saveReview: protectedProcedure
      .input(z.object({ weekStart: z.string() }))
      .mutation(async ({ ctx, input }) => {
        const weekStart = new Date(input.weekStart);
        const weekEnd = new Date(weekStart);
        weekEnd.setDate(weekEnd.getDate() + 7);
        const stats = await db.getWeeklyStats(ctx.user.id, weekStart, weekEnd);
        await db.saveReviewSnapshot({
          userId: ctx.user.id,
          weekStart,
          foundationCounts: stats.byFoundation,
          completionRate: stats.total > 0 ? stats.completed / stats.total : 0,
          totalTasks: stats.total,
          completedTasks: stats.completed,
          droppedTasks: stats.dropped,
          driftAnalysis: null,
        });
        return stats;
      }),

    export: protectedProcedure
      .input(z.object({ format: z.enum(["csv", "json"]) }))
      .query(async ({ ctx, input }) => {
        const allTasks = await db.getTasksByUser(ctx.user.id);
        if (input.format === "json") {
          return { data: JSON.stringify(allTasks, null, 2), contentType: "application/json" };
        }
        // CSV
        const headers = ["id", "title", "status", "archetype", "foundation", "subFoundation", "desireStage", "wisdomStage", "powerStage", "urgency", "priorityScore", "equationCanonical", "createdAt"];
        const rows = allTasks.map(t => [
          t.id, `"${(t.title || "").replace(/"/g, '""')}"`, t.status, t.archetype, t.foundationId, t.subFoundationId,
          t.desireStage, t.wisdomStage, t.powerStage, t.urgency, t.priorityScore, `"${(t.equationCanonical || "").replace(/"/g, '""')}"`, t.createdAt,
        ].join(","));
        return { data: [headers.join(","), ...rows].join("\n"), contentType: "text/csv" };
      }),
  }),
});

export type AppRouter = typeof appRouter;
