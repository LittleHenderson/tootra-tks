import { useState, useRef, useCallback, useEffect } from "react";
import { trpc } from "@/lib/trpc";
import { Button } from "@/components/ui/button";
import { Mic, MicOff, Loader2, Square } from "lucide-react";
import { toast } from "sonner";

interface VoiceRecorderProps {
  /** Called with the transcribed text when recording is complete */
  onTranscription: (text: string) => void;
  /** Optional: compact mode for inline use */
  compact?: boolean;
  /** Optional: className for the container */
  className?: string;
}

export default function VoiceRecorder({ onTranscription, compact = false, className = "" }: VoiceRecorderProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [duration, setDuration] = useState(0);
  const [audioLevel, setAudioLevel] = useState(0);
  const [isSupported, setIsSupported] = useState(true);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const transcribeMutation = trpc.voice.transcribe.useMutation({
    onSuccess: (data) => {
      setIsTranscribing(false);
      if (data.text && data.text.trim()) {
        onTranscription(data.text.trim());
        toast.success("Voice transcribed successfully");
      } else {
        toast.error("No speech detected. Please try again.");
      }
    },
    onError: (err) => {
      setIsTranscribing(false);
      toast.error("Transcription failed: " + err.message);
    },
  });

  // Check browser support
  useEffect(() => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setIsSupported(false);
    }
    // Check MediaRecorder support
    if (typeof MediaRecorder === "undefined") {
      setIsSupported(false);
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop());
      }
    };
  }, []);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 44100,
        },
      });
      streamRef.current = stream;

      // Set up audio analyser for waveform visualization
      const audioCtx = new AudioContext();
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      analyserRef.current = analyser;

      // Animate audio level
      const dataArray = new Uint8Array(analyser.frequencyBinCount);
      const updateLevel = () => {
        if (!analyserRef.current) return;
        analyserRef.current.getByteFrequencyData(dataArray);
        const avg = dataArray.reduce((sum, v) => sum + v, 0) / dataArray.length;
        setAudioLevel(avg / 255);
        animFrameRef.current = requestAnimationFrame(updateLevel);
      };
      updateLevel();

      // Determine best supported mime type
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : MediaRecorder.isTypeSupported("audio/webm")
        ? "audio/webm"
        : MediaRecorder.isTypeSupported("audio/mp4")
        ? "audio/mp4"
        : "audio/wav";

      const mediaRecorder = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = async () => {
        // Stop visualizer
        if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
        setAudioLevel(0);

        // Stop stream tracks
        stream.getTracks().forEach(t => t.stop());
        streamRef.current = null;

        // Build blob
        const blob = new Blob(chunksRef.current, { type: mimeType });
        if (blob.size === 0) {
          toast.error("No audio recorded. Please try again.");
          return;
        }

        // Check size
        const sizeMB = blob.size / (1024 * 1024);
        if (sizeMB > 16) {
          toast.error(`Recording is ${sizeMB.toFixed(1)}MB — maximum is 16MB. Try a shorter recording.`);
          return;
        }

        // Convert to base64
        setIsTranscribing(true);
        const reader = new FileReader();
        reader.onloadend = () => {
          const base64 = (reader.result as string).split(",")[1];
          if (!base64) {
            setIsTranscribing(false);
            toast.error("Failed to process audio");
            return;
          }
          const baseMime = mimeType.split(";")[0];
          transcribeMutation.mutate({
            audioBase64: base64,
            mimeType: baseMime,
          });
        };
        reader.readAsDataURL(blob);
      };

      mediaRecorder.start(250); // collect data every 250ms
      setIsRecording(true);
      setDuration(0);

      // Start timer
      timerRef.current = setInterval(() => {
        setDuration(d => d + 1);
      }, 1000);
    } catch (err: any) {
      if (err.name === "NotAllowedError") {
        toast.error("Microphone access denied. Please allow microphone permissions.");
      } else if (err.name === "NotFoundError") {
        toast.error("No microphone found. Please connect a microphone.");
      } else {
        toast.error("Could not start recording: " + (err.message || "Unknown error"));
      }
    }
  }, [transcribeMutation]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setIsRecording(false);
  }, []);

  const formatDuration = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  if (!isSupported) {
    return (
      <div className={`inline-flex items-center gap-1.5 text-xs text-muted-foreground/60 ${className}`}>
        <MicOff className="h-3 w-3" />
        <span>Voice not supported</span>
      </div>
    );
  }

  // Compact mode: just a mic button
  if (compact) {
    return (
      <div className={`inline-flex items-center gap-2 ${className}`}>
        {isTranscribing ? (
          <Button
            size="sm"
            variant="outline"
            disabled
            className="h-10 w-10 p-0 border-tks-gold/30"
          >
            <Loader2 className="h-4 w-4 animate-spin text-tks-gold" />
          </Button>
        ) : isRecording ? (
          <>
            {/* Waveform indicator */}
            <div className="flex items-center gap-0.5 h-6">
              {[...Array(5)].map((_, i) => (
                <div
                  key={i}
                  className="w-1 rounded-full bg-tks-flame transition-all duration-75"
                  style={{
                    height: `${Math.max(4, audioLevel * 24 * (0.5 + Math.random() * 0.5))}px`,
                  }}
                />
              ))}
            </div>
            <span className="text-xs text-tks-flame font-mono tabular-nums">{formatDuration(duration)}</span>
            <Button
              size="sm"
              variant="outline"
              onClick={stopRecording}
              className="h-10 w-10 p-0 border-tks-flame/50 text-tks-flame hover:bg-tks-flame/10 animate-pulse"
            >
              <Square className="h-4 w-4 fill-current" />
            </Button>
          </>
        ) : (
          <Button
            size="sm"
            variant="outline"
            onClick={startRecording}
            className="h-10 w-10 p-0 border-border hover:border-tks-gold/50 hover:text-tks-gold transition-colors"
            title="Record voice"
          >
            <Mic className="h-4 w-4" />
          </Button>
        )}
      </div>
    );
  }

  // Full mode: larger button with more feedback
  return (
    <div className={`flex items-center gap-3 ${className}`}>
      {isTranscribing ? (
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-tks-gold/10 border border-tks-gold/20">
          <Loader2 className="h-4 w-4 animate-spin text-tks-gold" />
          <span className="text-xs text-tks-gold">Transcribing...</span>
        </div>
      ) : isRecording ? (
        <div className="flex items-center gap-3 px-3 py-2 rounded-lg bg-tks-flame/10 border border-tks-flame/30">
          {/* Waveform bars */}
          <div className="flex items-center gap-0.5 h-8">
            {[...Array(8)].map((_, i) => (
              <div
                key={i}
                className="w-1 rounded-full bg-tks-flame transition-all duration-75"
                style={{
                  height: `${Math.max(4, audioLevel * 32 * (0.4 + Math.random() * 0.6))}px`,
                }}
              />
            ))}
          </div>
          <span className="text-sm text-tks-flame font-mono tabular-nums min-w-[3ch]">
            {formatDuration(duration)}
          </span>
          <Button
            size="sm"
            onClick={stopRecording}
            className="bg-tks-flame hover:bg-tks-flame/80 text-white h-8"
          >
            <Square className="h-3 w-3 fill-current mr-1" />
            Stop
          </Button>
        </div>
      ) : (
        <Button
          size="sm"
          variant="outline"
          onClick={startRecording}
          className="border-border hover:border-tks-gold/50 hover:text-tks-gold transition-colors"
          title="Record voice to capture a task"
        >
          <Mic className="h-4 w-4 mr-1.5" />
          Speak
        </Button>
      )}
    </div>
  );
}
