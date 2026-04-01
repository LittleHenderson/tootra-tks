import { TOKEN_TYPES } from "../../../shared/tks";
import type { TokenType } from "../../../shared/tks";

interface EquationToken {
  type: string;
  value: string;
  operator?: string;
}

export function EquationChip({ token }: { token: EquationToken }) {
  const tokenDef = TOKEN_TYPES[token.type as TokenType];
  if (!tokenDef) return <span className="text-xs text-muted-foreground">{token.type}({token.value})</span>;

  return (
    <span
      className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium border transition-all hover:scale-105"
      style={{
        backgroundColor: tokenDef.bgColor,
        borderColor: tokenDef.color + "40",
        color: tokenDef.color,
      }}
    >
      <span className="text-[10px]">{tokenDef.icon}</span>
      <span className="font-semibold">{token.type}</span>
      <span className="opacity-80">({token.value})</span>
    </span>
  );
}

export function OperatorChip({ operator }: { operator: string }) {
  if (!operator) return null;
  return (
    <span className="inline-flex items-center px-1.5 py-0.5 text-xs font-mono text-tks-gold/70">
      {operator}
    </span>
  );
}

export function EquationDisplay({ tokens }: { tokens: EquationToken[] }) {
  if (!tokens || tokens.length === 0) {
    return <span className="text-xs text-muted-foreground italic">No equation generated</span>;
  }

  return (
    <div className="flex flex-wrap items-center gap-1">
      {tokens.map((token, i) => (
        <span key={i} className="inline-flex items-center gap-0.5">
          {token.operator && <OperatorChip operator={token.operator} />}
          <EquationChip token={token} />
        </span>
      ))}
    </div>
  );
}

export function EquationCanonical({ equation }: { equation: string }) {
  if (!equation) return null;
  return (
    <code className="text-xs font-mono text-muted-foreground bg-secondary/50 px-2 py-1 rounded block overflow-x-auto">
      {equation}
    </code>
  );
}
