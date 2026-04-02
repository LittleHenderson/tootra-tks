import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { INVERSION_MODES, INVERSION_PRESETS } from "../../../shared/inversions";
import {
  ArrowRight,
  RotateCcw,
  Sparkles,
  Image as ImageIcon,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import { useState } from "react";
import { Streamdown } from "streamdown";

interface InversionResultProps {
  inversion: {
    id: number;
    modeIds: number[] | string;
    intensity: string;
    scope: string;
    presetId?: string | null;
    originalEquation?: string | null;
    originalNarrative?: string | null;
    invertedEquation?: string | null;
    invertedNarrative?: string | null;
    changeSummary?: any;
    invertedFoundationId?: number | null;
    invertedArchetype?: string | null;
    invertedDesireStage?: string | null;
    invertedWisdomStage?: string | null;
    invertedPowerStage?: string | null;
    scenarioPrompt?: string | null;
    imageUrl?: string | null;
    createdAt?: string | Date | null;
  };
  compact?: boolean;
}

export function InversionResult({ inversion, compact = false }: InversionResultProps) {
  const [showDetails, setShowDetails] = useState(!compact);
  const [showPrompt, setShowPrompt] = useState(false);

  // Parse modeIds if it's a string
  const modeIds: number[] = typeof inversion.modeIds === "string"
    ? JSON.parse(inversion.modeIds)
    : (inversion.modeIds || []);

  const modes = modeIds
    .map((id) => INVERSION_MODES.find((m) => m.id === id))
    .filter(Boolean);

  const preset = inversion.presetId
    ? INVERSION_PRESETS.find((p) => p.id === inversion.presetId)
    : null;

  // Parse changeSummary if it's a string
  const changeSummary = typeof inversion.changeSummary === "string"
    ? JSON.parse(inversion.changeSummary)
    : inversion.changeSummary;

  return (
    <Card className="border-purple-500/20 bg-gradient-to-br from-purple-950/20 to-transparent overflow-hidden">
      <CardContent className="p-4 space-y-4">
        {/* Header */}
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-2">
            <RotateCcw className="h-4 w-4 text-purple-400" />
            <span className="font-semibold text-sm text-purple-300">
              {preset ? (
                <>
                  <span className="mr-1">{preset.icon}</span>
                  {preset.name}
                </>
              ) : (
                "Custom Inversion"
              )}
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <Badge variant="outline" className="text-[10px] px-1.5 border-purple-500/30 text-purple-400">
              {inversion.intensity}
            </Badge>
            <Badge variant="outline" className="text-[10px] px-1.5 border-purple-500/30 text-purple-400">
              {inversion.scope}
            </Badge>
          </div>
        </div>

        {/* Mode badges */}
        <div className="flex flex-wrap gap-1">
          {modes.map((mode) =>
            mode ? (
              <Badge
                key={mode.id}
                variant="secondary"
                className="text-[10px] px-1.5 py-0"
              >
                {mode.name}
              </Badge>
            ) : null
          )}
        </div>

        {/* Equation Comparison */}
        {(inversion.originalEquation || inversion.invertedEquation) && (
          <div className="space-y-2">
            <button
              className="flex items-center gap-1 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors"
              onClick={() => setShowDetails(!showDetails)}
            >
              {showDetails ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
              Equation Transformation
            </button>

            {showDetails && (
              <div className="grid grid-cols-[1fr_auto_1fr] gap-3 items-start">
                {/* Original */}
                <div className="rounded-lg bg-background/50 border border-border/50 p-3">
                  <div className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1.5 font-medium">
                    Original
                  </div>
                  <div className="text-xs font-mono text-foreground/80 break-all">
                    {inversion.originalEquation || "—"}
                  </div>
                </div>

                {/* Arrow */}
                <div className="flex items-center justify-center pt-6">
                  <ArrowRight className="h-5 w-5 text-purple-400" />
                </div>

                {/* Inverted */}
                <div className="rounded-lg bg-purple-500/5 border border-purple-500/20 p-3">
                  <div className="text-[10px] uppercase tracking-wider text-purple-400 mb-1.5 font-medium">
                    Inverted
                  </div>
                  <div className="text-xs font-mono text-foreground/80 break-all">
                    {inversion.invertedEquation || "—"}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Change Summary */}
        {showDetails && changeSummary && (
          <div className="space-y-2">
            <h5 className="text-xs font-medium text-muted-foreground flex items-center gap-1">
              <Sparkles className="h-3 w-3" />
              What Changed
            </h5>
            {Array.isArray(changeSummary) ? (
              <div className="space-y-1.5">
                {changeSummary.map((change: any, i: number) => (
                  <div
                    key={i}
                    className="flex items-start gap-2 text-xs rounded-md bg-background/30 px-2.5 py-1.5"
                  >
                    <span className="text-purple-400 font-medium shrink-0">
                      {change.axis || change.field || `#${i + 1}`}:
                    </span>
                    <span className="text-muted-foreground">
                      {change.from && change.to
                        ? `${change.from} → ${change.to}`
                        : change.description || change.change || JSON.stringify(change)}
                    </span>
                  </div>
                ))}
              </div>
            ) : typeof changeSummary === "object" ? (
              <div className="text-xs text-muted-foreground bg-background/30 rounded-md p-2.5">
                <Streamdown>{JSON.stringify(changeSummary, null, 2)}</Streamdown>
              </div>
            ) : null}
          </div>
        )}

        {/* Inverted Narrative */}
        {showDetails && inversion.invertedNarrative && (
          <div className="space-y-1.5">
            <h5 className="text-xs font-medium text-muted-foreground">
              Inverted Narrative
            </h5>
            <div className="text-sm text-foreground/90 leading-relaxed bg-background/30 rounded-lg p-3">
              <Streamdown>{inversion.invertedNarrative}</Streamdown>
            </div>
          </div>
        )}

        {/* Inverted TKS Attributes */}
        {showDetails && (inversion.invertedArchetype || inversion.invertedDesireStage || inversion.invertedWisdomStage || inversion.invertedPowerStage) && (
          <div className="flex flex-wrap gap-2">
            {inversion.invertedArchetype && (
              <Badge variant="outline" className="text-[10px] border-amber-500/30 text-amber-400">
                Archetype: {inversion.invertedArchetype}
              </Badge>
            )}
            {inversion.invertedDesireStage && (
              <Badge variant="outline" className="text-[10px] border-red-500/30 text-red-400">
                D: {inversion.invertedDesireStage}
              </Badge>
            )}
            {inversion.invertedWisdomStage && (
              <Badge variant="outline" className="text-[10px] border-blue-500/30 text-blue-400">
                W: {inversion.invertedWisdomStage}
              </Badge>
            )}
            {inversion.invertedPowerStage && (
              <Badge variant="outline" className="text-[10px] border-emerald-500/30 text-emerald-400">
                P: {inversion.invertedPowerStage}
              </Badge>
            )}
          </div>
        )}

        {/* Generated Image */}
        {inversion.imageUrl && (
          <div className="space-y-1.5">
            <h5 className="text-xs font-medium text-muted-foreground flex items-center gap-1">
              <ImageIcon className="h-3 w-3" />
              Inverted Vision
            </h5>
            <div className="rounded-lg overflow-hidden border border-purple-500/20">
              <img
                src={inversion.imageUrl}
                alt="Inverted scenario vision"
                className="w-full aspect-[16/9] object-cover"
              />
            </div>
          </div>
        )}

        {/* Scenario Prompt (collapsible) */}
        {showDetails && inversion.scenarioPrompt && (
          <div>
            <button
              className="flex items-center gap-1 text-[10px] text-muted-foreground/60 hover:text-muted-foreground transition-colors"
              onClick={() => setShowPrompt(!showPrompt)}
            >
              {showPrompt ? <ChevronUp className="h-2.5 w-2.5" /> : <ChevronDown className="h-2.5 w-2.5" />}
              View scenario prompt
            </button>
            {showPrompt && (
              <div className="mt-1 text-[10px] text-muted-foreground/60 bg-background/20 rounded p-2 font-mono">
                {inversion.scenarioPrompt}
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
