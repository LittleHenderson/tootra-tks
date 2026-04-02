import { eq, and, desc, asc, sql, gte, lte, inArray, count } from "drizzle-orm";
import { drizzle } from "drizzle-orm/mysql2";
import { InsertUser, users, tasks, Task, InsertTask, powerInventory, InsertPowerInventoryItem, dependencies, InsertDependency, reviewSnapshots, InsertReviewSnapshot, scenarios, InsertScenario, inversions, InsertInversion } from "../drizzle/schema";
import { ENV } from './_core/env';

let _db: ReturnType<typeof drizzle> | null = null;

export async function getDb() {
  if (!_db && process.env.DATABASE_URL) {
    try {
      _db = drizzle(process.env.DATABASE_URL);
    } catch (error) {
      console.warn("[Database] Failed to connect:", error);
      _db = null;
    }
  }
  return _db;
}

// ── User Helpers ────────────────────────────────────────────
export async function upsertUser(user: InsertUser): Promise<void> {
  if (!user.openId) throw new Error("User openId is required for upsert");
  const db = await getDb();
  if (!db) { console.warn("[Database] Cannot upsert user: database not available"); return; }
  try {
    const values: InsertUser = { openId: user.openId };
    const updateSet: Record<string, unknown> = {};
    const textFields = ["name", "email", "loginMethod"] as const;
    type TextField = (typeof textFields)[number];
    const assignNullable = (field: TextField) => {
      const value = user[field];
      if (value === undefined) return;
      const normalized = value ?? null;
      values[field] = normalized;
      updateSet[field] = normalized;
    };
    textFields.forEach(assignNullable);
    if (user.lastSignedIn !== undefined) { values.lastSignedIn = user.lastSignedIn; updateSet.lastSignedIn = user.lastSignedIn; }
    if (user.role !== undefined) { values.role = user.role; updateSet.role = user.role; }
    else if (user.openId === ENV.ownerOpenId) { values.role = 'admin'; updateSet.role = 'admin'; }
    if (!values.lastSignedIn) values.lastSignedIn = new Date();
    if (Object.keys(updateSet).length === 0) updateSet.lastSignedIn = new Date();
    await db.insert(users).values(values).onDuplicateKeyUpdate({ set: updateSet });
  } catch (error) { console.error("[Database] Failed to upsert user:", error); throw error; }
}

export async function getUserByOpenId(openId: string) {
  const db = await getDb();
  if (!db) return undefined;
  const result = await db.select().from(users).where(eq(users.openId, openId)).limit(1);
  return result.length > 0 ? result[0] : undefined;
}

export async function updateUserMode(userId: number, mode: "life_os" | "work_sprint") {
  const db = await getDb();
  if (!db) return;
  await db.update(users).set({ mode }).where(eq(users.id, userId));
}

// ── Task Helpers ────────────────────────────────────────────
export async function createTask(data: InsertTask) {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  const result = await db.insert(tasks).values(data);
  const id = result[0].insertId;
  const rows = await db.select().from(tasks).where(eq(tasks.id, id)).limit(1);
  return rows[0];
}

export async function getTasksByUser(userId: number, status?: string) {
  const db = await getDb();
  if (!db) return [];
  const conditions = [eq(tasks.userId, userId)];
  if (status) conditions.push(eq(tasks.status, status as any));
  return db.select().from(tasks).where(and(...conditions)).orderBy(desc(tasks.priorityScore), desc(tasks.createdAt));
}

export async function getTaskById(taskId: number, userId: number) {
  const db = await getDb();
  if (!db) return undefined;
  const rows = await db.select().from(tasks).where(and(eq(tasks.id, taskId), eq(tasks.userId, userId))).limit(1);
  return rows[0];
}

export async function updateTask(taskId: number, userId: number, data: Partial<InsertTask>) {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  await db.update(tasks).set(data).where(and(eq(tasks.id, taskId), eq(tasks.userId, userId)));
  return getTaskById(taskId, userId);
}

export async function deleteTask(taskId: number, userId: number) {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  await db.delete(tasks).where(and(eq(tasks.id, taskId), eq(tasks.userId, userId)));
}

// ── Power Inventory Helpers ─────────────────────────────────
export async function getPowerInventory(taskId: number, userId: number) {
  const db = await getDb();
  if (!db) return [];
  return db.select().from(powerInventory).where(and(eq(powerInventory.taskId, taskId), eq(powerInventory.userId, userId)));
}

export async function upsertPowerItem(data: InsertPowerInventoryItem) {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  const existing = await db.select().from(powerInventory)
    .where(and(eq(powerInventory.taskId, data.taskId), eq(powerInventory.userId, data.userId), eq(powerInventory.category, data.category)))
    .limit(1);
  if (existing.length > 0) {
    await db.update(powerInventory).set({ status: data.status, notes: data.notes, suggestedSubtask: data.suggestedSubtask })
      .where(eq(powerInventory.id, existing[0].id));
    return { ...existing[0], ...data };
  } else {
    const result = await db.insert(powerInventory).values(data);
    return { id: result[0].insertId, ...data };
  }
}

// ── Dependencies Helpers ────────────────────────────────────
export async function getTaskDependencies(taskId: number, userId: number) {
  const db = await getDb();
  if (!db) return [];
  return db.select().from(dependencies).where(and(eq(dependencies.sourceTaskId, taskId), eq(dependencies.userId, userId)));
}

export async function createDependency(data: InsertDependency) {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  await db.insert(dependencies).values(data);
}

// ── Analytics Helpers ───────────────────────────────────────
export async function getFoundationHeatmap(userId: number) {
  const db = await getDb();
  if (!db) return [];
  const result = await db.select({
    foundationId: tasks.foundationId,
    subFoundationId: tasks.subFoundationId,
    total: count(),
    completed: sql<number>`SUM(CASE WHEN ${tasks.status} = 'done' THEN 1 ELSE 0 END)`,
    active: sql<number>`SUM(CASE WHEN ${tasks.status} IN ('inbox', 'scheduled', 'active') THEN 1 ELSE 0 END)`,
  }).from(tasks).where(eq(tasks.userId, userId)).groupBy(tasks.foundationId, tasks.subFoundationId);
  return result;
}

export async function getWeeklyStats(userId: number, weekStart: Date, weekEnd: Date) {
  const db = await getDb();
  if (!db) return { total: 0, completed: 0, dropped: 0, byFoundation: [] };
  const allTasks = await db.select().from(tasks)
    .where(and(eq(tasks.userId, userId), gte(tasks.createdAt, weekStart), lte(tasks.createdAt, weekEnd)));
  const total = allTasks.length;
  const completed = allTasks.filter(t => t.status === "done").length;
  const dropped = allTasks.filter(t => t.status === "dropped").length;
  const byFoundation: Record<number, { total: number; completed: number }> = {};
  for (const t of allTasks) {
    const fId = t.foundationId || 0;
    if (!byFoundation[fId]) byFoundation[fId] = { total: 0, completed: 0 };
    byFoundation[fId].total++;
    if (t.status === "done") byFoundation[fId].completed++;
  }
  return { total, completed, dropped, byFoundation };
}

export async function saveReviewSnapshot(data: InsertReviewSnapshot) {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  await db.insert(reviewSnapshots).values(data);
}

export async function getReviewSnapshots(userId: number, limit = 12) {
  const db = await getDb();
  if (!db) return [];
  return db.select().from(reviewSnapshots).where(eq(reviewSnapshots.userId, userId)).orderBy(desc(reviewSnapshots.weekStart)).limit(limit);
}

// ── User Photo ─────────────────────────────────────────────
export async function updateUserPhoto(userId: number, photoUrl: string) {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  await db.update(users).set({ photoUrl }).where(eq(users.id, userId));
}

export async function getUserPhoto(userId: number): Promise<string | null> {
  const db = await getDb();
  if (!db) return null;
  const rows = await db.select({ photoUrl: users.photoUrl }).from(users).where(eq(users.id, userId)).limit(1);
  return rows[0]?.photoUrl || null;
}

// ── Scenario Helpers ───────────────────────────────────────
export async function createScenario(data: InsertScenario) {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  const result = await db.insert(scenarios).values(data);
  const id = result[0].insertId;
  const rows = await db.select().from(scenarios).where(eq(scenarios.id, id)).limit(1);
  return rows[0];
}

export async function getScenariosForTask(taskId: number, userId: number) {
  const db = await getDb();
  if (!db) return [];
  return db.select().from(scenarios).where(and(eq(scenarios.taskId, taskId), eq(scenarios.userId, userId))).orderBy(desc(scenarios.createdAt));
}

// ── Inversion Helpers ─────────────────────────────────────────
export async function createInversion(data: InsertInversion) {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  const result = await db.insert(inversions).values(data);
  const id = result[0].insertId;
  const rows = await db.select().from(inversions).where(eq(inversions.id, id)).limit(1);
  return rows[0];
}

export async function getInversionsForTask(taskId: number, userId: number) {
  const db = await getDb();
  if (!db) return [];
  return db.select().from(inversions).where(and(eq(inversions.taskId, taskId), eq(inversions.userId, userId))).orderBy(desc(inversions.createdAt));
}

export async function getInversionById(id: number, userId: number) {
  const db = await getDb();
  if (!db) return null;
  const rows = await db.select().from(inversions).where(and(eq(inversions.id, id), eq(inversions.userId, userId))).limit(1);
  return rows[0] || null;
}
