import { useState, useRef, useCallback } from "react";
import { trpc } from "@/lib/trpc";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";
import {
  Camera,
  Sparkles,
  ImagePlus,
  Loader2,
  Upload,
  Eye,
  SplitSquareHorizontal,
  ThumbsUp,
  ThumbsDown,
  X,
  User,
  Wand2,
} from "lucide-react";

interface ScenarioizeProps {
  taskId: number;
  taskTitle: string;
  autoOpen?: boolean;
}

export default function Scenarioize({ taskId, taskTitle, autoOpen = false }: ScenarioizeProps) {
  const [isOpen, setIsOpen] = useState(autoOpen);
  const [mode, setMode] = useState<"single" | "dual">("single");
  const [singleType, setSingleType] = useState<"positive" | "negative">("positive");
  const [isGenerating, setIsGenerating] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const photoQuery = trpc.scenario.getPhoto.useQuery(undefined, { enabled: isOpen });
  const scenariosQuery = trpc.scenario.listForTask.useQuery({ taskId }, { enabled: isOpen });
  const uploadPhotoMut = trpc.scenario.uploadPhoto.useMutation();
  const generateMut = trpc.scenario.generate.useMutation();
  const utils = trpc.useUtils();

  const handlePhotoUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (file.size > 10 * 1024 * 1024) {
      toast.error("Photo must be under 10MB");
      return;
    }

    setIsUploading(true);
    try {
      const reader = new FileReader();
      reader.onload = async () => {
        const base64 = (reader.result as string).split(",")[1];
        await uploadPhotoMut.mutateAsync({
          photoBase64: base64,
          mimeType: file.type,
        });
        utils.scenario.getPhoto.invalidate();
        toast.success("Photo uploaded! It will be used as reference in your scenarios.");
      };
      reader.readAsDataURL(file);
    } catch (err) {
      toast.error("Failed to upload photo");
    } finally {
      setIsUploading(false);
    }
  }, [uploadPhotoMut, utils]);

  const handleGenerate = useCallback(async () => {
    setIsGenerating(true);
    try {
      const result = await generateMut.mutateAsync({
        taskId,
        mode,
        type: singleType,
      });
      utils.scenario.listForTask.invalidate({ taskId });
      const count = result.scenarios.length;
      toast.success(`Generated ${count} scenario${count > 1 ? "s" : ""}!`);
    } catch (err: any) {
      toast.error(err?.message || "Failed to generate scenario");
    } finally {
      setIsGenerating(false);
    }
  }, [generateMut, taskId, mode, singleType, utils]);

  if (!isOpen) {
    return (
      <Button
        variant="outline"
        size="sm"
        onClick={() => setIsOpen(true)}
        className="gap-2 border-amber-500/30 text-amber-400 hover:bg-amber-500/10 hover:text-amber-300"
      >
        <Wand2 className="h-4 w-4" />
        Scenarioize
      </Button>
    );
  }

  const existingScenarios = scenariosQuery.data || [];
  const positiveScenarios = existingScenarios.filter(s => s.type === "positive");
  const negativeScenarios = existingScenarios.filter(s => s.type === "negative");

  return (
    <Card className="border-amber-500/20 bg-gradient-to-br from-amber-950/20 to-transparent">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-amber-400 text-lg">
            <Sparkles className="h-5 w-5" />
            Scenarioize — Vision Engine
          </CardTitle>
          <Button variant="ghost" size="icon" onClick={() => setIsOpen(false)} className="h-8 w-8 text-muted-foreground">
            <X className="h-4 w-4" />
          </Button>
        </div>
        <p className="text-sm text-muted-foreground">
          See a realistic AI-generated vision of what happens if you follow through — or don't.
        </p>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Photo Upload Section */}
        <div className="flex items-center gap-4 p-3 rounded-lg bg-background/50 border border-border/50">
          <div className="relative">
            {photoQuery.data?.photoUrl ? (
              <img
                src={photoQuery.data.photoUrl}
                alt="Your photo"
                className="h-14 w-14 rounded-full object-cover border-2 border-amber-500/40"
              />
            ) : (
              <div className="h-14 w-14 rounded-full bg-muted flex items-center justify-center border-2 border-dashed border-muted-foreground/30">
                <User className="h-6 w-6 text-muted-foreground" />
              </div>
            )}
          </div>
          <div className="flex-1">
            <p className="text-sm font-medium">
              {photoQuery.data?.photoUrl ? "Your reference photo" : "Upload a photo of yourself"}
            </p>
            <p className="text-xs text-muted-foreground">
              {photoQuery.data?.photoUrl
                ? "This photo will be used as a reference when generating your scenarios."
                : "Optional — helps the AI create more personalized scenario images."}
            </p>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
            className="gap-1.5"
          >
            {isUploading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Upload className="h-3.5 w-3.5" />}
            {photoQuery.data?.photoUrl ? "Change" : "Upload"}
          </Button>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/png,image/webp"
            className="hidden"
            onChange={handlePhotoUpload}
          />
        </div>

        {/* Generation Controls */}
        <div className="space-y-3">
          <div className="flex items-center gap-3">
            <span className="text-sm font-medium text-foreground">Mode:</span>
            <Tabs value={mode} onValueChange={(v) => setMode(v as "single" | "dual")} className="flex-1">
              <TabsList className="grid w-full grid-cols-2 h-9">
                <TabsTrigger value="single" className="gap-1.5 text-xs">
                  <Eye className="h-3.5 w-3.5" />
                  Single Image
                </TabsTrigger>
                <TabsTrigger value="dual" className="gap-1.5 text-xs">
                  <SplitSquareHorizontal className="h-3.5 w-3.5" />
                  Dual Vision
                </TabsTrigger>
              </TabsList>
            </Tabs>
          </div>

          {mode === "single" && (
            <div className="flex items-center gap-3">
              <span className="text-sm font-medium text-foreground">Show me:</span>
              <div className="flex gap-2">
                <Button
                  variant={singleType === "positive" ? "default" : "outline"}
                  size="sm"
                  onClick={() => setSingleType("positive")}
                  className={`gap-1.5 ${singleType === "positive" ? "bg-emerald-600 hover:bg-emerald-700" : ""}`}
                >
                  <ThumbsUp className="h-3.5 w-3.5" />
                  If I do this
                </Button>
                <Button
                  variant={singleType === "negative" ? "default" : "outline"}
                  size="sm"
                  onClick={() => setSingleType("negative")}
                  className={`gap-1.5 ${singleType === "negative" ? "bg-red-600 hover:bg-red-700" : ""}`}
                >
                  <ThumbsDown className="h-3.5 w-3.5" />
                  If I don't
                </Button>
              </div>
            </div>
          )}

          <Button
            onClick={handleGenerate}
            disabled={isGenerating}
            className="w-full gap-2 bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-700 hover:to-orange-700 text-white"
          >
            {isGenerating ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Generating {mode === "dual" ? "visions" : "vision"}... (10-20s)
              </>
            ) : (
              <>
                <ImagePlus className="h-4 w-4" />
                Generate {mode === "dual" ? "Dual Vision" : "Scenario Image"}
              </>
            )}
          </Button>
        </div>

        {/* Generated Scenarios Display */}
        {existingScenarios.length > 0 && (
          <div className="space-y-4 pt-2">
            <h4 className="text-sm font-semibold text-foreground flex items-center gap-2">
              <Camera className="h-4 w-4 text-amber-400" />
              Generated Visions
            </h4>

            {/* If we have both positive and negative, show side by side */}
            {positiveScenarios.length > 0 && negativeScenarios.length > 0 ? (
              <div className="grid grid-cols-2 gap-3">
                {/* Positive */}
                <div className="space-y-2">
                  <Badge className="bg-emerald-600/20 text-emerald-400 border-emerald-500/30">
                    <ThumbsUp className="h-3 w-3 mr-1" />
                    If you follow through
                  </Badge>
                  <div className="rounded-lg overflow-hidden border border-emerald-500/20">
                    <img
                      src={positiveScenarios[0].imageUrl!}
                      alt="Positive outcome"
                      className="w-full aspect-[4/3] object-cover"
                    />
                  </div>
                  <p className="text-xs text-muted-foreground">{positiveScenarios[0].description}</p>
                </div>
                {/* Negative */}
                <div className="space-y-2">
                  <Badge className="bg-red-600/20 text-red-400 border-red-500/30">
                    <ThumbsDown className="h-3 w-3 mr-1" />
                    If you don't
                  </Badge>
                  <div className="rounded-lg overflow-hidden border border-red-500/20">
                    <img
                      src={negativeScenarios[0].imageUrl!}
                      alt="Negative consequence"
                      className="w-full aspect-[4/3] object-cover"
                    />
                  </div>
                  <p className="text-xs text-muted-foreground">{negativeScenarios[0].description}</p>
                </div>
              </div>
            ) : (
              /* Single scenario display */
              <div className="space-y-3">
                {existingScenarios.slice(0, 4).map((sc) => (
                  <div key={sc.id} className="space-y-2">
                    <Badge className={sc.type === "positive"
                      ? "bg-emerald-600/20 text-emerald-400 border-emerald-500/30"
                      : "bg-red-600/20 text-red-400 border-red-500/30"
                    }>
                      {sc.type === "positive" ? <ThumbsUp className="h-3 w-3 mr-1" /> : <ThumbsDown className="h-3 w-3 mr-1" />}
                      {sc.type === "positive" ? "If you follow through" : "If you don't"}
                    </Badge>
                    <div className={`rounded-lg overflow-hidden border ${
                      sc.type === "positive" ? "border-emerald-500/20" : "border-red-500/20"
                    }`}>
                      <img
                        src={sc.imageUrl!}
                        alt={`${sc.type} scenario`}
                        className="w-full aspect-[16/9] object-cover"
                      />
                    </div>
                    <p className="text-xs text-muted-foreground">{sc.description}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
