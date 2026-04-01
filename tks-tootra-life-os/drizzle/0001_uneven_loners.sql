CREATE TABLE `dependencies` (
	`id` int AUTO_INCREMENT NOT NULL,
	`userId` int NOT NULL,
	`sourceTaskId` int NOT NULL,
	`targetTaskId` int NOT NULL,
	`type` enum('requires','blocks','enables','caused_by') NOT NULL DEFAULT 'requires',
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `dependencies_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `power_inventory` (
	`id` int AUTO_INCREMENT NOT NULL,
	`taskId` int NOT NULL,
	`userId` int NOT NULL,
	`category` enum('time','money','tools','access','body','skill','connections','environment') NOT NULL,
	`status` enum('have','lack','partial') NOT NULL DEFAULT 'lack',
	`notes` text,
	`suggestedSubtask` text,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `power_inventory_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `review_snapshots` (
	`id` int AUTO_INCREMENT NOT NULL,
	`userId` int NOT NULL,
	`weekStart` timestamp NOT NULL,
	`foundationCounts` json,
	`completionRate` float,
	`totalTasks` int,
	`completedTasks` int,
	`droppedTasks` int,
	`driftAnalysis` json,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `review_snapshots_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `tasks` (
	`id` int AUTO_INCREMENT NOT NULL,
	`userId` int NOT NULL,
	`title` varchar(500) NOT NULL,
	`rawInput` text,
	`description` text,
	`status` enum('inbox','scheduled','active','done','dropped') NOT NULL DEFAULT 'inbox',
	`archetype` enum('execution','prep','maintenance','repair','study','audit') DEFAULT 'execution',
	`desireStage` varchar(32) DEFAULT 'D_SUNRISE',
	`wisdomStage` varchar(32) DEFAULT 'W_LEARNING',
	`powerStage` varchar(32) DEFAULT 'P_BIRTH',
	`dwpDiagnosis` text,
	`foundationId` int,
	`subFoundationId` varchar(8),
	`mappingConfidence` float,
	`scheduledAt` timestamp,
	`dueAt` timestamp,
	`durationMinutes` int,
	`recurrence` varchar(128),
	`priorityScore` float DEFAULT 0,
	`urgency` enum('low','medium','high','critical') DEFAULT 'medium',
	`equationCanonical` text,
	`equationAst` json,
	`completedAt` timestamp,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `tasks_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
ALTER TABLE `users` ADD `mode` enum('life_os','work_sprint') DEFAULT 'life_os' NOT NULL;