import { useState, useMemo } from "react";
import {
  INVERSION_MODES,
  INVERSION_PRESETS,
  INVERSION_LAYERS,
  INVERSION_INTENSITIES,
  INVERSION_SCOPES,
  type InversionMode,
  type InversionLayer,
  type InversionIntensity,
  type InversionScope,
  type InversionPreset,
} from "../../../shared/inversions";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  RotateCcw,
  Zap,
  Flame,
  Waves,
  ChevronDown,
  ChevronUp,
  Sparkles,
  Image as ImageIcon,
  Loader2,
  Info,
  X,
} from "lucide-react";

interface InversionDialProps {
  taskId: number;
  onPerform: (config: {
    modeIds: number[];
    intensity: InversionIntensity;
    scope: InversionScope;
    presetId?: string;
    generateImage: boolean;
  }) => void;
  isLoading?: boolean;
}

export function InversionDial({ taskId, onPerform, isLoading }: InversionDialProps) {
  const [selectedModeIds, setSelectedModeIds] = useState<number[]>([]);
  const [intensity, setIntensity] = useState<InversionIntensity>("medium");
  const [scope, setScope] = useState<InversionScope>("equation");
  const [generateImage, setGenerateImage] = useState(false);
  const [expandedLayer, setExpandedLayer] = useState<InversionLayer | null>("surface");
  const [activeTab, setActiveTab] = useState<"modes" | "presets">("presets");
  const [showAdvanced, setShowAdvanced] = useState(false);

  const modesByLayer = useMemo(() => {
    const grouped: Record<InversionLayer, InversionMode[]> = {
      surface: [],
      deep_structural: [],
      meta_layer: [],
      special: [],
    };
    INVERSION_MODES.forEach((m) => grouped[m.layer].push(m));
    return grouped;
  }, []);

  const toggleMode = (id: number) => {
    setSelectedModeIds((prev) => {
      if (prev.includes(id)) return prev.filter((x) => x !== id);
      if (prev.length >= 5) return prev; // Max 5 modes
      return [...prev, id];
    });
  };

  const applyPreset = (preset: InversionPreset) => {
    onPerform({
      modeIds: preset.modeIds,
      intensity: preset.intensity,
      scope: preset.scope,
      presetId: preset.id,
      generateImage,
    });
  };

  const performCustom = () => {
    if (selectedModeIds.length === 0) return;
    onPerform({
      modeIds: selectedModeIds,
      intensity,
      scope,
      generateImage,
    });
  };

  const intensityIcons: Record<InversionIntensity, React.ReactNode> = {
    soft: <Waves className="h-4 w-4" />,
    medium: <Zap className="h-4 w-4" />,
    hard: <Flame className="h-4 w-4" />,
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-purple-500 to-red-500">
          <RotateCcw className="h-5 w-5 text-white" />
        </div>
        <div>
          <h3 className="text-lg font-bold">Inversion Engine</h3>
          <p className="text-xs text-muted-foreground">
            29 modes across 4 layers — see your scenario from every angle
          </p>
        </div>
      </div>

      {/* Tabs: Presets vs Custom */}
      <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as any)}>
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="presets">
            <Sparkles className="mr-2 h-4 w-4" />
            Preset Recipes
          </TabsTrigger>
          <TabsTrigger value="modes">
            <RotateCcw className="mr-2 h-4 w-4" />
            Custom Mix
          </TabsTrigger>
        </TabsList>

        {/* ═══ PRESETS TAB ═══ */}
        <TabsContent value="presets" className="space-y-3 mt-3">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {INVERSION_PRESETS.map((preset) => (
              <Card
                key={preset.id}
                className="cursor-pointer transition-all hover:shadow-lg hover:scale-[1.02] border-2 border-transparent hover:border-purple-500/30"
                onClick={() => !isLoading && applyPreset(preset)}
              >
                <CardContent className="p-4">
                  <div className="flex items-start gap-3">
                    <span className="text-2xl">{preset.icon}</span>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <h4 className="font-semibold text-sm">{preset.name}</h4>
                        <Badge
                          variant="outline"
                          className="text-[10px] px-1.5 py-0"
                          style={{ borderColor: preset.color, color: preset.color }}
                        >
                          {preset.intensity}
                        </Badge>
                      </div>
                      <p className="text-xs text-muted-foreground leading-relaxed">
                        {preset.description}
                      </p>
                      <div className="flex flex-wrap gap-1 mt-2">
                        {preset.modeIds.map((id) => {
                          const mode = INVERSION_MODES.find((m) => m.id === id);
                          return mode ? (
                            <Badge
                              key={id}
                              variant="secondary"
                              className="text-[9px] px-1 py-0"
                            >
                              {mode.name}
                            </Badge>
                          ) : null;
                        })}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Image generation toggle */}
          <div className="flex items-center gap-2 px-1">
            <button
              onClick={() => setGenerateImage(!generateImage)}
              className={`flex items-center gap-2 rounded-lg border px-3 py-2 text-xs transition-colors ${
                generateImage
                  ? "border-purple-500 bg-purple-500/10 text-purple-400"
                  : "border-border text-muted-foreground hover:border-purple-500/50"
              }`}
            >
              <ImageIcon className="h-3.5 w-3.5" />
              Generate inverted vision image
            </button>
          </div>
        </TabsContent>

        {/* ═══ CUSTOM MIX TAB ═══ */}
        <TabsContent value="modes" className="space-y-4 mt-3">
          {/* Layer Accordion */}
          {(Object.keys(modesByLayer) as InversionLayer[]).map((layer) => {
            const layerInfo = INVERSION_LAYERS[layer];
            const modes = modesByLayer[layer];
            const isExpanded = expandedLayer === layer;
            const selectedInLayer = modes.filter((m) =>
              selectedModeIds.includes(m.id)
            ).length;

            return (
              <div key={layer} className="rounded-lg border overflow-hidden">
                <button
                  className="flex w-full items-center justify-between p-3 text-left hover:bg-accent/50 transition-colors"
                  onClick={() =>
                    setExpandedLayer(isExpanded ? null : layer)
                  }
                >
                  <div className="flex items-center gap-3">
                    <div
                      className="h-3 w-3 rounded-full"
                      style={{ backgroundColor: layerInfo.color }}
                    />
                    <div>
                      <span className="font-semibold text-sm">
                        {layerInfo.label}
                      </span>
                      <span className="text-xs text-muted-foreground ml-2">
                        {modes.length} modes
                      </span>
                    </div>
                    {selectedInLayer > 0 && (
                      <Badge variant="default" className="text-[10px] px-1.5">
                        {selectedInLayer} selected
                      </Badge>
                    )}
                  </div>
                  {isExpanded ? (
                    <ChevronUp className="h-4 w-4 text-muted-foreground" />
                  ) : (
                    <ChevronDown className="h-4 w-4 text-muted-foreground" />
                  )}
                </button>

                {isExpanded && (
                  <div className="border-t p-3 space-y-2">
                    <p className="text-xs text-muted-foreground mb-3">
                      {layerInfo.description}
                    </p>
                    <div className="grid grid-cols-1 gap-2">
                      {modes.map((mode) => {
                        const isSelected = selectedModeIds.includes(mode.id);
                        return (
                          <TooltipProvider key={mode.id}>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <button
                                  className={`flex items-start gap-3 rounded-lg border p-3 text-left transition-all ${
                                    isSelected
                                      ? "border-purple-500 bg-purple-500/10"
                                      : "border-border hover:border-purple-500/30 hover:bg-accent/30"
                                  } ${
                                    !isSelected && selectedModeIds.length >= 5
                                      ? "opacity-50 cursor-not-allowed"
                                      : "cursor-pointer"
                                  }`}
                                  onClick={() => toggleMode(mode.id)}
                                  disabled={
                                    !isSelected && selectedModeIds.length >= 5
                                  }
                                >
                                  <div
                                    className={`mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded border text-xs font-bold ${
                                      isSelected
                                        ? "border-purple-500 bg-purple-500 text-white"
                                        : "border-muted-foreground/30"
                                    }`}
                                  >
                                    {isSelected ? "✓" : mode.id}
                                  </div>
                                  <div className="min-w-0">
                                    <div className="font-medium text-sm">
                                      {mode.name}
                                    </div>
                                    <div className="text-xs text-muted-foreground mt-0.5">
                                      {mode.shortDescription}
                                    </div>
                                  </div>
                                </button>
                              </TooltipTrigger>
                              <TooltipContent
                                side="right"
                                className="max-w-xs"
                              >
                                <p className="text-xs">{mode.description}</p>
                                <div className="mt-2 text-[10px] text-muted-foreground">
                                  <strong>Example:</strong>{" "}
                                  {mode.example.before} → {mode.example.after}
                                </div>
                              </TooltipContent>
                            </Tooltip>
                          </TooltipProvider>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            );
          })}

          {/* Intensity Selector */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Intensity</label>
            <div className="flex gap-2">
              {(Object.keys(INVERSION_INTENSITIES) as InversionIntensity[]).map(
                (level) => {
                  const info = INVERSION_INTENSITIES[level];
                  return (
                    <button
                      key={level}
                      className={`flex flex-1 items-center justify-center gap-2 rounded-lg border px-3 py-2.5 text-sm transition-all ${
                        intensity === level
                          ? "border-purple-500 bg-purple-500/10 text-purple-400 font-medium"
                          : "border-border text-muted-foreground hover:border-purple-500/30"
                      }`}
                      onClick={() => setIntensity(level)}
                    >
                      {intensityIcons[level]}
                      {info.label}
                    </button>
                  );
                }
              )}
            </div>
          </div>

          {/* Advanced: Scope */}
          <div>
            <button
              className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
              onClick={() => setShowAdvanced(!showAdvanced)}
            >
              <Info className="h-3 w-3" />
              Advanced options
              {showAdvanced ? (
                <ChevronUp className="h-3 w-3" />
              ) : (
                <ChevronDown className="h-3 w-3" />
              )}
            </button>

            {showAdvanced && (
              <div className="mt-3 space-y-3 rounded-lg border p-3">
                <div className="space-y-2">
                  <label className="text-xs font-medium">Scope</label>
                  <div className="flex flex-wrap gap-1.5">
                    {(Object.keys(INVERSION_SCOPES) as InversionScope[]).map(
                      (s) => {
                        const info = INVERSION_SCOPES[s];
                        return (
                          <button
                            key={s}
                            className={`rounded-md border px-2.5 py-1 text-xs transition-all ${
                              scope === s
                                ? "border-purple-500 bg-purple-500/10 text-purple-400"
                                : "border-border text-muted-foreground hover:border-purple-500/30"
                            }`}
                            onClick={() => setScope(s)}
                          >
                            {info.label}
                          </button>
                        );
                      }
                    )}
                  </div>
                </div>

                {/* Image generation toggle */}
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setGenerateImage(!generateImage)}
                    className={`flex items-center gap-2 rounded-lg border px-3 py-2 text-xs transition-colors ${
                      generateImage
                        ? "border-purple-500 bg-purple-500/10 text-purple-400"
                        : "border-border text-muted-foreground hover:border-purple-500/50"
                    }`}
                  >
                    <ImageIcon className="h-3.5 w-3.5" />
                    Generate inverted vision image
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Selected modes summary + Perform button */}
          {selectedModeIds.length > 0 && (
            <div className="rounded-lg border border-purple-500/30 bg-purple-500/5 p-3 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium">
                  {selectedModeIds.length} mode{selectedModeIds.length > 1 ? "s" : ""} selected
                </span>
                <button
                  className="text-xs text-muted-foreground hover:text-foreground"
                  onClick={() => setSelectedModeIds([])}
                >
                  <X className="h-3 w-3" />
                </button>
              </div>
              <div className="flex flex-wrap gap-1">
                {selectedModeIds.map((id) => {
                  const mode = INVERSION_MODES.find((m) => m.id === id);
                  return mode ? (
                    <Badge
                      key={id}
                      variant="secondary"
                      className="text-xs cursor-pointer hover:bg-destructive/20"
                      onClick={() => toggleMode(id)}
                    >
                      {mode.name} ×
                    </Badge>
                  ) : null;
                })}
              </div>
              <Button
                className="w-full bg-gradient-to-r from-purple-600 to-red-600 hover:from-purple-700 hover:to-red-700"
                onClick={performCustom}
                disabled={isLoading}
              >
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Inverting...
                  </>
                ) : (
                  <>
                    <RotateCcw className="mr-2 h-4 w-4" />
                    Perform Inversion
                  </>
                )}
              </Button>
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}
