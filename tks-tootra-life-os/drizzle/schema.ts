import { int, mysqlEnum, mysqlTable, text, timestamp, varchar, json, float, boolean } from "drizzle-orm/mysql-core";

// ── Users ───────────────────────────────────────────────────
export const users = mysqlTable("users", {
  id: int("id").autoincrement().primaryKey(),
  openId: varchar("openId", { length: 64 }).notNull().unique(),
  name: text("name"),
  email: varchar("email", { length: 320 }),
  loginMethod: varchar("loginMethod", { length: 64 }),
  role: mysqlEnum("role", ["user", "admin"]).default("user").notNull(),
  mode: mysqlEnum("mode", ["life_os", "work_sprint"]).default("life_os").notNull(),
  photoUrl: text("photoUrl"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
  lastSignedIn: timestamp("lastSignedIn").defaultNow().notNull(),
});

export type User = typeof users.$inferSelect;
export type InsertUser = typeof users.$inferInsert;

// ── Tasks ───────────────────────────────────────────────────
export const tasks = mysqlTable("tasks", {
  id: int("id").autoincrement().primaryKey(),
  userId: int("userId").notNull(),
  title: varchar("title", { length: 500 }).notNull(),
  rawInput: text("rawInput"),
  description: text("description"),
  status: mysqlEnum("status", ["inbox", "scheduled", "active", "done", "dropped"]).default("inbox").notNull(),
  archetype: mysqlEnum("archetype", ["execution", "prep", "maintenance", "repair", "study", "audit"]).default("execution"),
  // D/W/P Lifecycle
  desireStage: varchar("desireStage", { length: 32 }).default("D_SUNRISE"),
  wisdomStage: varchar("wisdomStage", { length: 32 }).default("W_LEARNING"),
  powerStage: varchar("powerStage", { length: 32 }).default("P_BIRTH"),
  dwpDiagnosis: text("dwpDiagnosis"),
  // Foundation mapping
  foundationId: int("foundationId"),
  subFoundationId: varchar("subFoundationId", { length: 8 }),
  mappingConfidence: float("mappingConfidence"),
  // Scheduling
  scheduledAt: timestamp("scheduledAt"),
  dueAt: timestamp("dueAt"),
  durationMinutes: int("durationMinutes"),
  recurrence: varchar("recurrence", { length: 128 }),
  // Priority
  priorityScore: float("priorityScore").default(0),
  urgency: mysqlEnum("urgency", ["low", "medium", "high", "critical"]).default("medium"),
  // Equation reference
  equationCanonical: text("equationCanonical"),
  equationAst: json("equationAst"),
  // Metadata
  completedAt: timestamp("completedAt"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
});

export type Task = typeof tasks.$inferSelect;
export type InsertTask = typeof tasks.$inferInsert;

// ── Power Inventory ─────────────────────────────────────────
export const powerInventory = mysqlTable("power_inventory", {
  id: int("id").autoincrement().primaryKey(),
  taskId: int("taskId").notNull(),
  userId: int("userId").notNull(),
  category: mysqlEnum("category", ["time", "money", "tools", "access", "body", "skill", "connections", "environment"]).notNull(),
  status: mysqlEnum("status", ["have", "lack", "partial"]).default("lack").notNull(),
  notes: text("notes"),
  suggestedSubtask: text("suggestedSubtask"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
});

export type PowerInventoryItem = typeof powerInventory.$inferSelect;
export type InsertPowerInventoryItem = typeof powerInventory.$inferInsert;

// ── Dependencies ────────────────────────────────────────────
export const dependencies = mysqlTable("dependencies", {
  id: int("id").autoincrement().primaryKey(),
  userId: int("userId").notNull(),
  sourceTaskId: int("sourceTaskId").notNull(),
  targetTaskId: int("targetTaskId").notNull(),
  type: mysqlEnum("type", ["requires", "blocks", "enables", "caused_by"]).default("requires").notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});

export type Dependency = typeof dependencies.$inferSelect;
export type InsertDependency = typeof dependencies.$inferInsert;

// ── Review Snapshots (for weekly review) ────────────────────
export const reviewSnapshots = mysqlTable("review_snapshots", {
  id: int("id").autoincrement().primaryKey(),
  userId: int("userId").notNull(),
  weekStart: timestamp("weekStart").notNull(),
  foundationCounts: json("foundationCounts"),
  completionRate: float("completionRate"),
  totalTasks: int("totalTasks"),
  completedTasks: int("completedTasks"),
  droppedTasks: int("droppedTasks"),
  driftAnalysis: json("driftAnalysis"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});

export type ReviewSnapshot = typeof reviewSnapshots.$inferSelect;
export type InsertReviewSnapshot = typeof reviewSnapshots.$inferInsert;

// ── Scenarios (Scenarioize Vision) ─────────────────────────
export const scenarios = mysqlTable("scenarios", {
  id: int("id").autoincrement().primaryKey(),
  taskId: int("taskId").notNull(),
  userId: int("userId").notNull(),
  type: mysqlEnum("type", ["positive", "negative"]).notNull(),
  prompt: text("prompt"),
  imageUrl: text("imageUrl"),
  description: text("description"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});

export type Scenario = typeof scenarios.$inferSelect;
export type InsertScenario = typeof scenarios.$inferInsert;
