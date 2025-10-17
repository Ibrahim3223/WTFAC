# -*- coding: utf-8 -*-
"""
Main orchestrator - coordinates all modules.
"""
import os
import time
import shutil
import tempfile
import pathlib
import datetime
import hashlib
from typing import Optional, Dict, Any

from autoshorts.config import settings
from autoshorts.content.gemini_client import GeminiClient
from autoshorts.content.quality_scorer import QualityScorer
from autoshorts.tts.edge_handler import TTSHandler
from autoshorts.video.pexels_client import PexelsClient
from autoshorts.video.downloader import VideoDownloader
from autoshorts.video.segment_maker import SegmentMaker
from autoshorts.captions.renderer import CaptionRenderer
from autoshorts.audio.bgm_manager import BGMManager
from autoshorts.upload.youtube_uploader import YouTubeUploader
from autoshorts.state.novelty_guard import NoveltyGuard
from autoshorts.state.state_guard import StateGuard
from autoshorts.utils.ffmpeg_utils import run, ffprobe_duration


class ShortsOrchestrator:
    """Orchestrate the complete shorts generation pipeline."""
    
    def __init__(self):
        """Initialize all modules."""
        self.channel = settings.CHANNEL_NAME
        self.temp_dir = None
        
        # Modules
        self.gemini = GeminiClient()
        self.quality_scorer = QualityScorer()
        self.tts = TTSHandler()
        self.pexels = PexelsClient()
        self.downloader = VideoDownloader()
        self.segment_maker = SegmentMaker()
        self.caption_renderer = CaptionRenderer()
        self.bgm_manager = BGMManager()
        
        if settings.UPLOAD_TO_YT:
            self.uploader = YouTubeUploader()
        
        # State
        self.novelty_guard = NoveltyGuard(
            state_dir=settings.STATE_DIR,
            window_days=settings.ENTITY_COOLDOWN_DAYS
        )
        self.state_guard = StateGuard(channel=self.channel)
        
        print(f"🚀 Orchestrator ready: {self.channel}")
    
    def run(self) -> Optional[str]:
        """Execute full pipeline. Returns YouTube video ID or None."""
        self.temp_dir = tempfile.mkdtemp(prefix="shorts_")
        
        try:
            # Phase 1: Content
            print("\n📝 Phase 1: Content generation...")
            content = self._generate_content()
            if not content:
                return None
            
            # Phase 2: TTS
            print("\n🎤 Phase 2: Text-to-speech...")
            audio_segments = self._generate_tts(content['sentences'])
            if not audio_segments:
                return None
            
            # Phase 3: Video
            print("\n🎬 Phase 3: Video production...")
            video_path = self._produce_video(
                audio_segments,
                content['focus'],
                content['search_terms']
            )
            if not video_path:
                return None
            
            # Phase 4: Upload
            if settings.UPLOAD_TO_YT:
                print("\n📤 Phase 4: Upload...")
                video_id = self._upload(video_path, content)
                print(f"✅ Success! https://youtube.com/watch?v={video_id}")
                return video_id
            else:
                print(f"⏭️ Upload disabled. Saved: {video_path}")
                return None
                
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            raise
        finally:
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
    
    def _generate_content(self) -> Optional[Dict[str, Any]]:
        """Generate content with quality and novelty checks."""
        attempts = 0
        best = None
        best_score = -1.0
        
        while attempts < settings.NOVELTY_RETRIES:
            attempts += 1
            print(f"   Attempt {attempts}/{settings.NOVELTY_RETRIES}")
            
            # Generate
            content = self.gemini.generate(
                topic=settings.TOPIC,
                mode=settings.MODE,
                lang=settings.LANG
            )
            
            # Quality check
            scores = self.quality_scorer.score(content['sentences'], content['title'])
            overall = scores['overall']
            
            print(f"   Q={scores['quality']:.1f} V={scores['viral']:.1f} R={scores['retention']:.1f} → {overall:.1f}")
            
            # Novelty check
            decision = self.novelty_guard.check_novelty(
                channel=self.channel,
                title=content['title'],
                script=" ".join(content['sentences']),
                search_term=content.get('selected_term'),
                category=settings.MODE,
                mode=settings.MODE,
                lang=settings.LANG
            )
            
            if not decision.ok:
                print(f"   ⚠️ {decision.reason}")
                continue
            
            # Accept if good enough
            if overall >= settings.MIN_OVERALL_SCORE:
                best = content
                break
            
            # Track best
            if overall > best_score:
                best = content
                best_score = overall
        
        if best:
            # Register
            self.novelty_guard.register_item(
                channel=self.channel,
                title=best['title'],
                script=" ".join(best['sentences']),
                search_term=best.get('selected_term'),
                category=settings.MODE,
                mode=settings.MODE,
                lang=settings.LANG,
                topic=best['topic']
            )
        
        return best
    
    def _generate_tts(self, sentences):
        """Generate TTS for all sentences."""
        audio_segments = []
        
        for i, text in enumerate(sentences):
            wav_path = os.path.join(self.temp_dir, f"sent_{i:02d}.wav")
            duration, words = self.tts.synthesize(text, wav_path)
            
            audio_segments.append({
                'text': text,
                'wav_path': wav_path,
                'duration': duration,
                'words': words
            })
            
            print(f"   {i+1}/{len(sentences)}: {duration:.2f}s")
        
        return audio_segments
    
    def _produce_video(self, audio_segments, focus, search_terms):
        """Produce final video."""
        # Get video pool
        print("   🔎 Pexels search...")
        pool = self.pexels.build_pool(focus, search_terms, len(audio_segments))
        
        if not pool:
            return None
        
        # Download
        print("   ⬇️ Download...")
        downloads = self.downloader.download(pool, self.temp_dir)
        
        # Create segments with captions
        print("   🎨 Segments...")
        segments = []
        video_files = list(downloads.values())[:len(audio_segments)]
        
        for i, audio in enumerate(audio_segments):
            video_src = video_files[i] if i < len(video_files) else video_files[-1]
            
            segment = self.segment_maker.create(
                video_src=video_src,
                duration=audio['duration'],
                temp_dir=self.temp_dir,
                index=i
            )
            
            segment_with_caption = self.caption_renderer.render(
                video_path=segment,
                text=audio['text'],
                words=audio['words'],
                is_hook=(i == 0),
                temp_dir=self.temp_dir
            )
            
            segments.append(segment_with_caption)
        
        # Concat video
        video_concat = os.path.join(self.temp_dir, "video.mp4")
        self._concat_videos(segments, video_concat)
        
        # Concat audio
        audio_concat = os.path.join(self.temp_dir, "audio.wav")
        self._concat_audios([a['wav_path'] for a in audio_segments], audio_concat)
        
        # BGM
        if settings.BGM_ENABLE:
            print("   🎧 BGM...")
            audio_dur = ffprobe_duration(audio_concat)
            audio_concat = self.bgm_manager.add_bgm(audio_concat, audio_dur, self.temp_dir)
        
        # Mux
        print("   🔄 Mux...")
        ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        final = os.path.join(settings.OUT_DIR, f"{self.channel}_{ts}.mp4")
        os.makedirs(settings.OUT_DIR, exist_ok=True)
        
        self._mux(video_concat, audio_concat, final)
        
        return final
    
    def _concat_videos(self, files, output):
        """Concatenate video files."""
        inputs = []
        filters = []
        
        for i, p in enumerate(files):
            inputs += ["-i", p]
            filters.append(f"[{i}:v]setsar=1,fps={settings.TARGET_FPS},settb=AVTB[v{i}]")
        
        filtergraph = ";".join(filters) + ";" + "".join(f"[v{i}]" for i in range(len(files))) + f"concat=n={len(files)}:v=1:a=0[v]"
        
        run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            *inputs,
            "-filter_complex", filtergraph,
            "-map", "[v]",
            "-r", str(settings.TARGET_FPS), "-vsync", "cfr",
            "-c:v", "libx264", "-preset", "medium", "-crf", str(settings.CRF_VISUAL),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            output
        ])
    
    def _concat_audios(self, files, output):
        """Concatenate audio files."""
        lst = output.replace(".wav", ".txt")
        with open(lst, "w") as f:
            for p in files:
                f.write(f"file '{p}'\n")
        
        run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", lst,
            "-c", "copy",
            output
        ])
        
        os.unlink(lst)
    
    def _mux(self, video, audio, output):
        """Mux video and audio."""
        run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", video, "-i", audio,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "256k",
            "-movflags", "+faststart",
            output
        ])
    
    def _upload(self, video_path, content):
        """Upload to YouTube."""
        metadata = {
            'title': content['title'],
            'description': content['description'],
            'tags': content['tags'],
            'privacy': settings.VISIBILITY,
            'defaultLanguage': settings.LANG,
            'defaultAudioLanguage': settings.LANG
        }
        
        video_id = self.uploader.upload(video_path, metadata)
        
        # Mark uploaded
        self.state_guard.mark_uploaded(
            entity=content.get('focus', ''),
            script_text=" ".join(content['sentences']),
            content_hash=hashlib.md5(content['title'].encode()).hexdigest(),
            video_path=video_path,
            title=content['title']
        )
        
        return video_id
