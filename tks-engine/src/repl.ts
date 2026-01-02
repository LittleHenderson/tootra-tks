/**
 * TKS Core Engine v0 - REPL
 *
 * Simple Read-Eval-Print Loop for the TKS system.
 * Supports: let, eval, show state, help, show foundations/acquisitions, set foundation/acquisition, and bare expressions.
 */

import * as readline from 'readline';
import { parseCommand, Command, ParseError } from './parser';
import { evaluate, valueToString, EvaluationError } from './evaluator';
import {
  TksState,
  createDefaultState,
  TksValue,
  FoundationId,
  AcquisitionId,
  isFoundationId,
  isAcquisitionId,
  ALL_FOUNDATION_IDS,
  ALL_ACQUISITION_IDS,
} from './types';
import { TksExpression } from './ast';
import { checkPrereqs } from './rpm';
import { coachForFoundation } from './coach';

// ============================================================================
// REPL CLASS
// ============================================================================

export class TksRepl {
  private state: TksState;
  private rl: readline.Interface | null = null;

  constructor(initialState?: TksState) {
    this.state = initialState ?? createDefaultState();
  }

  /**
   * Starts the interactive REPL
   */
  start(): void {
    this.rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });

    console.log("TKS Core Engine v0 REPL");
    console.log("Type 'help' for available commands, or enter a TKS expression.");
    console.log("");

    this.prompt();
  }

  private prompt(): void {
    this.rl!.question("tks> ", (input) => {
      const trimmed = input.trim();

      if (trimmed === "" || trimmed === "exit" || trimmed === "quit") {
        if (trimmed === "" ) {
          this.prompt();
          return;
        }
        console.log("Goodbye!");
        this.rl!.close();
        return;
      }

      this.processLine(trimmed);
      this.prompt();
    });
  }

  /**
   * Processes a single line of input (for testing or programmatic use)
   */
  processLine(input: string): string[] {
    const output: string[] = [];

    const trimmed = input.trim();
    const coachPrefix = "coach ";
    if (trimmed === "coach" || trimmed.startsWith(coachPrefix)) {
      const arg = trimmed === "coach" ? "" : trimmed.slice(coachPrefix.length).trim();
      if (!isFoundationId(arg)) {
        output.push(`Unknown foundation id: ${arg}`);
      } else {
        const foundationId = arg as FoundationId;
        const suggestion = coachForFoundation(foundationId, this.state);

        output.push(`Coach report for ${foundationId} – ${suggestion.foundationMeta.label}:`);
        output.push(`  Activation: ${suggestion.currentActivation.toFixed(2)}`);
        output.push(
          `  RPM required: ${suggestion.rpm.required.join(", ") || "(none)"}`
        );
        output.push(
          `  RPM missing: ${suggestion.missingAcquisitions.join(", ") || "(none)"}`
        );
        output.push("  Suggested patterns:");
        if (suggestion.exampleExpressions.length === 0) {
          output.push("    (none)");
        } else {
          for (const expr of suggestion.exampleExpressions) {
            output.push(`    ${expr}`);
          }
        }
        output.push(`  Narrative: ${suggestion.narrative}`);
      }

      for (const line of output) {
        console.log(line);
      }
      return output;
    }

    // Handle state sugar commands before going through the parser
    const handled = this.tryHandleStateCommand(trimmed, output);
    if (handled) {
      for (const line of output) {
        console.log(line);
      }
      return output;
    }

    try {
      const command = parseCommand(input);
      this.executeCommand(command, output);
    } catch (error) {
      if (error instanceof ParseError) {
        output.push(`Parse error: ${error.message}`);
      } else if (error instanceof EvaluationError) {
        output.push(`Evaluation error: ${error.message}`);
      } else if (error instanceof Error) {
        output.push(`Error: ${error.message}`);
      } else {
        output.push(`Unknown error: ${error}`);
      }
    }

    // Print output to console if in interactive mode
    for (const line of output) {
      console.log(line);
    }

    return output;
  }

  private executeCommand(command: Command, output: string[]): void {
    switch (command.kind) {
      case "let":
        this.executeLet(command.name, command.expr, output);
        break;

      case "eval":
        this.executeEval(command.expr, output);
        break;

      case "expr":
        this.executeEval(command.expr, output);
        break;

      case "showState":
        this.executeShowState(output);
        break;

      case "help":
        this.executeHelp(output);
        break;

      case "coach":
        this.executeCoach(command.id, output);
        break;
    }
  }

  /**
   * Handles REPL-only state commands:
   * - set foundation <id> <value>
   * - set acquisition <n> <true|false>
   * - show foundations
   * - show acquisitions
   */
  private tryHandleStateCommand(input: string, output: string[]): boolean {
    const trimmed = input.trim();
    const lower = trimmed.toLowerCase();

    if (lower.startsWith("set foundation ")) {
      const parts = trimmed.split(/\s+/);
      if (parts.length < 4) {
        output.push("Error: usage: set foundation <id> <value>");
        return true;
      }

      const id = parts[2];
      const valueStr = parts.slice(3).join(" ");
      const parsed = parseFloat(valueStr);

      if (!isFoundationId(id)) {
        output.push(`Error: invalid foundation ID '${id}'. Use 1a-7d.`);
        return true;
      }

      if (Number.isNaN(parsed)) {
        output.push(`Error: value must be a number. Got '${valueStr}'.`);
        return true;
      }

      const clamped = Math.max(0, Math.min(1, parsed));
      if (clamped !== parsed) {
        output.push(`Value ${parsed} out of range. Clamped to ${clamped}.`);
      }

      this.state.foundations[id] = clamped;
      output.push(`Set foundation ${id} = ${clamped}`);
      return true;
    }

    if (lower.startsWith("set acquisition ")) {
      const parts = trimmed.split(/\s+/);
      if (parts.length < 4) {
        output.push("Error: usage: set acquisition <id> <true|false>");
        return true;
      }

      const idNum = parseInt(parts[2], 10);
      const boolStr = parts[3].toLowerCase();

      if (!isAcquisitionId(idNum)) {
        output.push(`Error: invalid acquisition ID '${parts[2]}'. Use 1-22.`);
        return true;
      }

      const boolVal = this.parseBoolean(boolStr);
      if (boolVal === null) {
        output.push("Error: value must be true|false (true/t/1/yes/y or false/f/0/no/n).");
        return true;
      }

      this.state.acquisitions[idNum as AcquisitionId] = boolVal;
      output.push(`Set acquisition ${idNum} = ${boolVal}`);
      return true;
    }

    if (lower === "show foundations") {
      output.push("Foundations:");
      const lines: string[] = [];
      for (const id of ALL_FOUNDATION_IDS.slice().sort()) {
        const val = this.state.foundations[id];
        if (val > 0) {
          lines.push(`  ${id}: ${val}`);
        }
      }
      if (lines.length === 0) {
        output.push("  (none active)");
      } else {
        output.push(...lines);
      }
      return true;
    }

    if (lower === "show acquisitions") {
      output.push("Acquisitions:");
      const lines: string[] = [];
      for (const id of ALL_ACQUISITION_IDS.slice().sort((a, b) => a - b)) {
        if (this.state.acquisitions[id]) {
          lines.push(`  ${id}: true`);
        }
      }
      if (lines.length === 0) {
        output.push("  (none active)");
      } else {
        output.push(...lines);
      }
      return true;
    }

    return false;
  }

  private parseBoolean(value: string): boolean | null {
    const trueVals = ["true", "t", "1", "yes", "y"];
    const falseVals = ["false", "f", "0", "no", "n"];
    if (trueVals.includes(value)) {
      return true;
    }
    if (falseVals.includes(value)) {
      return false;
    }
    return null;
  }

  private executeLet(name: string, expr: TksExpression, output: string[]): void {
    const result = evaluate(expr, this.state);

    // Update state with new variable
    this.state = {
      ...result.state,
      vars: {
        ...result.state.vars,
        [name]: result.value,
      },
    };

    // Print trace if there are steps
    if (result.trace.length > 0) {
      output.push("Trace:");
      for (const line of result.trace) {
        output.push(`  ${line}`);
      }
    }

    output.push(`${name} = ${valueToString(result.value)}`);
  }

  private executeEval(expr: TksExpression, output: string[]): void {
    const result = evaluate(expr, this.state);
    this.state = result.state;

    // Print trace
    if (result.trace.length > 0) {
      output.push("Trace:");
      for (const line of result.trace) {
        output.push(`  ${line}`);
      }
    }

    // RPM prerequisite diagnostics for foundation goals
    if (typeof result.value === "string" && isFoundationId(result.value)) {
      const rpm = checkPrereqs(result.value, this.state);
      if (rpm.required.length > 0) {
        if (rpm.missing.length > 0) {
          output.push(`RPM check for goal ${rpm.goal}:`);
          output.push(`  Required acquisitions: ${rpm.required.join(", ")}`);
          output.push(`  Missing acquisitions: ${rpm.missing.join(", ")}`);
        } else {
          output.push(`RPM check for goal ${rpm.goal}: all required acquisitions present.`);
        }
      } else {
        // Optional informational line; keep minimal noise.
        output.push(`RPM check for goal ${rpm.goal}: no prerequisites defined in v1.`);
      }
    }

    output.push(`Result: ${valueToString(result.value)}`);
  }

  private executeCoach(id: string, output: string[]): void {
    if (!isFoundationId(id)) {
      output.push(`Unknown foundation id: ${id}`);
      return;
    }

    const suggestion = coachForFoundation(id, this.state);

    output.push(`Coach report for ${suggestion.goalFoundation} – ${suggestion.foundationMeta.label}`);
    output.push(`  Activation: ${suggestion.currentActivation.toFixed(2)}`);
    output.push(
      `  RPM required: ${
        suggestion.rpm.required.length > 0 ? suggestion.rpm.required.join(", ") : "(none)"
      }`
    );
    output.push(
      `  RPM missing: ${
        suggestion.missingAcquisitions.length > 0 ? suggestion.missingAcquisitions.join(", ") : "(none)"
      }`
    );
    output.push("  Suggested patterns:");
    if (suggestion.exampleExpressions.length === 0) {
      output.push("    (none)");
    } else {
      for (const expr of suggestion.exampleExpressions) {
        output.push(`    ${expr}`);
      }
    }
    output.push(`  Narrative: ${suggestion.narrative}`);
  }

  private executeShowState(output: string[]): void {
    output.push("=== TKS State ===");
    output.push("");

    output.push(`Current Noetic: ν${this.state.currentNoetic}`);
    output.push(`Current Element: ${this.state.currentElement ?? "(none)"}`);
    output.push("");

    // Show active foundations (those with value > 0)
    const activeFoundations: string[] = [];
    for (const [id, value] of Object.entries(this.state.foundations)) {
      if (value > 0) {
        activeFoundations.push(`${id}: ${value.toFixed(2)}`);
      }
    }
    if (activeFoundations.length > 0) {
      output.push("Active Foundations:");
      for (const f of activeFoundations) {
        output.push(`  ${f}`);
      }
    } else {
      output.push("Active Foundations: (none)");
    }
    output.push("");

    // Show acquired acquisitions
    const acquired: number[] = [];
    for (const [id, value] of Object.entries(this.state.acquisitions)) {
      if (value) {
        acquired.push(parseInt(id, 10));
      }
    }
    if (acquired.length > 0) {
      output.push(`Acquired: ${acquired.join(", ")}`);
    } else {
      output.push("Acquired: (none)");
    }
    output.push("");

    // Show variables
    const varNames = Object.keys(this.state.vars);
    if (varNames.length > 0) {
      output.push("Variables:");
      for (const name of varNames) {
        output.push(`  ${name} = ${valueToString(this.state.vars[name])}`);
      }
    } else {
      output.push("Variables: (none)");
    }
  }

  private executeHelp(output: string[]): void {
    output.push("=== TKS REPL Commands ===");
    output.push("");
    output.push("Commands:");
    output.push("  let <name> = <expr>   Define a variable");
    output.push("  eval <expr>           Evaluate an expression");
    output.push("  show state            Show current TKS state");
    output.push("  show foundations      Show active foundations (>0)");
    output.push("  show acquisitions     Show acquired items (true)");
    output.push("  set foundation <id> <value>    Set foundation activation (0-1, clamps)");
    output.push("  set acquisition <id> <true|false>    Set acquisition flag");
    output.push("  coach <foundationId>  Show TKS coach suggestions for a foundation");
    output.push("  help                  Show this help message");
    output.push("  exit / quit           Exit the REPL");
    output.push("");
    output.push("Expression Syntax:");
    output.push("  Foundations:  1a, 2b, 3c, 4d, ... 7d");
    output.push("  Ideas:        A1-A10, B1-B10, C1-C10, D1-D10");
    output.push("  Integers:     0, 1, 42, ...");
    output.push("  Variables:    any identifier (a-z, A-Z, _, 0-9)");
    output.push("");
    output.push("Operators (precedence high to low):");
    output.push("  !              NOT (unary)");
    output.push("  &              AND");
    output.push("  |              OR");
    output.push("  ->             Implication");
    output.push("");
    output.push("Noetic Terms:");
    output.push("  <expr>^N       Apply noetic N to expression (N = 0-9)");
    output.push("  Example:       3d^2, health^7");
    output.push("");
    output.push("Fractals:");
    output.push("  <<N:M:...>>(expr)   Apply noetic sequence to expression");
    output.push("  Example:            <<1:4:7>>(3d)");
    output.push("");
    output.push("Examples:");
    output.push("  3d                          Evaluate foundation 3d");
    output.push("  let health = 3d             Assign 3d to 'health'");
    output.push("  eval <<1:4:7>>(health)      Apply fractal to variable");
    output.push("  1 & 0                       Boolean AND (result: false)");
    output.push("  A1 -> B2                    Implication structure");
  }

  /**
   * Gets the current state (for testing)
   */
  getState(): TksState {
    return this.state;
  }

  /**
   * Sets the state (for testing)
   */
  setState(state: TksState): void {
    this.state = state;
  }
}

// ============================================================================
// MAIN ENTRY POINT
// ============================================================================

if (require.main === module) {
  const repl = new TksRepl();
  repl.start();
}

export { TksRepl as default };
