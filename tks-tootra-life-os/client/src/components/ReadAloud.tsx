import { useState, useEffect, useRef, useCallback } from "react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Volume2, VolumeX, Pause, Play, Square, ChevronDown, Settings2 } from "lucide-react";
import { toast } from "sonner";

interface ReadAloudProps {
  /** The text content to read aloud */
  text: string;
  /** Optional label shown next to the button */
  label?: string;
  /** Compact mode: just an icon button */
  compact?: boolean;
  /** Optional className */
  className?: string;
}

const STORAGE_KEY = "tks-tts-voice";
const RATE_KEY = "tks-tts-rate";

export default function ReadAloud({ text, label, compact = false, className = "" }: ReadAloudProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [selectedVoiceName, setSelectedVoiceName] = useState<string>(() => {
    return localStorage.getItem(STORAGE_KEY) || "";
  });
  const [rate, setRate] = useState<number>(() => {
    const saved = localStorage.getItem(RATE_KEY);
    return saved ? parseFloat(saved) : 1.0;
  });
  const [isSupported, setIsSupported] = useState(true);
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);

  // Check support and load voices
  useEffect(() => {
    if (!("speechSynthesis" in window)) {
      setIsSupported(false);
      return;
    }

    const loadVoices = () => {
      const available = speechSynthesis.getVoices();
      if (available.length > 0) {
        setVoices(available);
        // If no voice selected yet, pick a good English default
        if (!selectedVoiceName) {
          const english = available.filter(v => v.lang.startsWith("en"));
          const preferred = english.find(v => v.name.includes("Google") || v.name.includes("Natural")) || english[0] || available[0];
          if (preferred) {
            setSelectedVoiceName(preferred.name);
            localStorage.setItem(STORAGE_KEY, preferred.name);
          }
        }
      }
    };

    loadVoices();
    speechSynthesis.addEventListener("voiceschanged", loadVoices);
    return () => {
      speechSynthesis.removeEventListener("voiceschanged", loadVoices);
    };
  }, [selectedVoiceName]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (speechSynthesis.speaking) {
        speechSynthesis.cancel();
      }
    };
  }, []);

  const selectVoice = useCallback((name: string) => {
    setSelectedVoiceName(name);
    localStorage.setItem(STORAGE_KEY, name);
    // If currently playing, restart with new voice
    if (isPlaying) {
      speechSynthesis.cancel();
      setIsPlaying(false);
      setIsPaused(false);
    }
  }, [isPlaying]);

  const changeRate = useCallback((newRate: number) => {
    setRate(newRate);
    localStorage.setItem(RATE_KEY, newRate.toString());
    if (isPlaying) {
      speechSynthesis.cancel();
      setIsPlaying(false);
      setIsPaused(false);
    }
  }, [isPlaying]);

  const speak = useCallback(() => {
    if (!text.trim()) {
      toast.error("Nothing to read aloud");
      return;
    }

    if (isPaused) {
      speechSynthesis.resume();
      setIsPaused(false);
      return;
    }

    // Cancel any ongoing speech
    speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = rate;
    utterance.pitch = 1.0;

    // Set voice
    const voice = voices.find(v => v.name === selectedVoiceName);
    if (voice) utterance.voice = voice;

    utterance.onstart = () => {
      setIsPlaying(true);
      setIsPaused(false);
    };
    utterance.onend = () => {
      setIsPlaying(false);
      setIsPaused(false);
    };
    utterance.onerror = (e) => {
      // "interrupted" is normal when user stops manually
      if (e.error !== "interrupted") {
        toast.error("Speech failed: " + e.error);
      }
      setIsPlaying(false);
      setIsPaused(false);
    };

    utteranceRef.current = utterance;
    speechSynthesis.speak(utterance);
  }, [text, voices, selectedVoiceName, rate, isPaused]);

  const pause = useCallback(() => {
    speechSynthesis.pause();
    setIsPaused(true);
  }, []);

  const stop = useCallback(() => {
    speechSynthesis.cancel();
    setIsPlaying(false);
    setIsPaused(false);
  }, []);

  if (!isSupported) return null;

  // Group voices by language for the dropdown
  const englishVoices = voices.filter(v => v.lang.startsWith("en"));
  const otherVoices = voices.filter(v => !v.lang.startsWith("en"));

  const rates = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0];

  if (compact) {
    return (
      <div className={`inline-flex items-center gap-1 ${className}`}>
        {isPlaying ? (
          <>
            <Button
              size="sm"
              variant="outline"
              onClick={isPaused ? speak : pause}
              className="h-7 w-7 p-0 border-tks-blue/30 text-tks-blue"
              title={isPaused ? "Resume" : "Pause"}
            >
              {isPaused ? <Play className="h-3 w-3" /> : <Pause className="h-3 w-3" />}
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={stop}
              className="h-7 w-7 p-0 border-destructive/30 text-destructive"
              title="Stop"
            >
              <Square className="h-3 w-3 fill-current" />
            </Button>
          </>
        ) : (
          <Button
            size="sm"
            variant="outline"
            onClick={speak}
            className="h-7 w-7 p-0 border-border hover:border-tks-blue/50 hover:text-tks-blue transition-colors"
            title="Read aloud"
          >
            <Volume2 className="h-3 w-3" />
          </Button>
        )}
      </div>
    );
  }

  // Full mode with voice/rate controls
  return (
    <div className={`flex items-center gap-2 ${className}`}>
      {isPlaying ? (
        <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-tks-blue/10 border border-tks-blue/20">
          <Volume2 className="h-3.5 w-3.5 text-tks-blue animate-pulse" />
          <span className="text-xs text-tks-blue font-medium">Reading...</span>
          <Button
            size="sm"
            variant="ghost"
            onClick={isPaused ? speak : pause}
            className="h-6 w-6 p-0 text-tks-blue hover:text-tks-blue"
          >
            {isPaused ? <Play className="h-3 w-3" /> : <Pause className="h-3 w-3" />}
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={stop}
            className="h-6 w-6 p-0 text-destructive hover:text-destructive"
          >
            <Square className="h-3 w-3 fill-current" />
          </Button>
        </div>
      ) : (
        <Button
          size="sm"
          variant="outline"
          onClick={speak}
          className="border-border hover:border-tks-blue/50 hover:text-tks-blue transition-colors text-xs"
        >
          <Volume2 className="h-3.5 w-3.5 mr-1.5" />
          {label || "Read Aloud"}
        </Button>
      )}

      {/* Settings dropdown */}
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            size="sm"
            variant="ghost"
            className="h-7 w-7 p-0 text-muted-foreground hover:text-foreground"
            title="Voice settings"
          >
            <Settings2 className="h-3.5 w-3.5" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-64 max-h-80 overflow-y-auto">
          {/* Speed control */}
          <div className="px-2 py-1.5 text-xs font-medium text-muted-foreground">Speed</div>
          {rates.map(r => (
            <DropdownMenuItem
              key={r}
              onClick={() => changeRate(r)}
              className={`text-xs cursor-pointer ${rate === r ? "text-tks-gold font-medium" : ""}`}
            >
              {r === 1.0 ? "Normal (1x)" : `${r}x`}
              {rate === r && " \u2713"}
            </DropdownMenuItem>
          ))}

          {/* Voice selection */}
          {englishVoices.length > 0 && (
            <>
              <div className="px-2 py-1.5 text-xs font-medium text-muted-foreground mt-1 border-t border-border pt-2">
                English Voices
              </div>
              {englishVoices.slice(0, 15).map(v => (
                <DropdownMenuItem
                  key={v.name}
                  onClick={() => selectVoice(v.name)}
                  className={`text-xs cursor-pointer ${selectedVoiceName === v.name ? "text-tks-gold font-medium" : ""}`}
                >
                  <span className="truncate">{v.name}</span>
                  {selectedVoiceName === v.name && <span className="ml-auto shrink-0"> \u2713</span>}
                </DropdownMenuItem>
              ))}
            </>
          )}

          {otherVoices.length > 0 && (
            <>
              <div className="px-2 py-1.5 text-xs font-medium text-muted-foreground mt-1 border-t border-border pt-2">
                Other Voices
              </div>
              {otherVoices.slice(0, 10).map(v => (
                <DropdownMenuItem
                  key={v.name}
                  onClick={() => selectVoice(v.name)}
                  className={`text-xs cursor-pointer ${selectedVoiceName === v.name ? "text-tks-gold font-medium" : ""}`}
                >
                  <span className="truncate">{v.name} ({v.lang})</span>
                  {selectedVoiceName === v.name && <span className="ml-auto shrink-0"> \u2713</span>}
                </DropdownMenuItem>
              ))}
            </>
          )}
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
}
