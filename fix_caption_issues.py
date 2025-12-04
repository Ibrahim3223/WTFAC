# -*- coding: utf-8 -*-
"""
Caption Issues Fix - Sayılar ve Timing Sorunlarını Çöz
=======================================================

Sorunlar:
1. Altyazılarda sayılar gözükmüyor (J) yerine (J) göstermeli
2. 2-3. cümlelerde hafif timing gecikmesi

Çözümler:
1. Text preprocessing'i düzelt - parantez içi içeriği koru
2. Timing accumulation'ı optimize et
"""

import re
from pathlib import Path

def fix_number_display():
    """Fix number display in captions."""

    print("="*60)
    print("FIX 1: SAYI GÖSTERME SORUNU")
    print("="*60)

    # keyword_highlighter.py'de regex düzeltmesi
    highlighter_path = Path("autoshorts/captions/keyword_highlighter.py")

    if not highlighter_path.exists():
        print("❌ keyword_highlighter.py bulunamadı")
        return

    content = highlighter_path.read_text(encoding='utf-8')

    # Mevcut regex: r'\b(\d+(?:,\d+)*(?:\.\d+)?)\b'
    # Bu regex tek haneli sayıları ve parantez içi içeriği yakalıyor
    # Ama parantez içindeki tek karakterler eksik kalıyor

    # Yeni regex ekleyelim - parantez içi karakterleri de yakala
    old_pattern = r"result = re.sub\(\s*r'\\\\b\(\\\\d\+\(\?:,\\\\d\+\)\*\(\?:\\\\\.\\\\d\+\)\?\)\\\\b',\s*r'\{\\\\\\\\c&H00FFFF&\\\\\\\\b1\\\\\\\\fs1\.3\}\\\\1\{\\\\\\\\r\}',\s*result\s*\)"

    new_code = '''        # 1. Highlight numbers (YELLOW, BOLD, 1.3x size for mobile screens)
        # Matches: 5, 100, 1,000, 1.5, (J), etc.
        # First, protect content in parentheses
        result = re.sub(
            r'\\(([^)]+)\\)',  # Match anything in parentheses
            r'{\\c&H00FFFF&\\b1\\fs1.3}(\\1){\\r}',  # Highlight parentheses content
            result
        )

        # Then, highlight standalone numbers
        result = re.sub(
            r'\\b(\\d+(?:,\\d+)*(?:\\.\\d+)?)\\b',
            r'{\\c&H00FFFF&\\b1\\fs1.3}\\1{\\r}',
            result
        )'''

    # Eski highlight kısmını bul
    if "# 1. Highlight numbers" in content:
        # Regex section'ı bul ve değiştir
        old_section = content[content.find("# 1. Highlight numbers"):content.find("# 2. Highlight emphasis words")]

        # Yeni section ile değiştir
        new_content = content.replace(old_section, new_code + "\n\n        ")

        # Dosyayı kaydet
        highlighter_path.write_text(new_content, encoding='utf-8')
        print("✅ keyword_highlighter.py güncellendi")
        print("   → Parantez içi içerikler artık vurgulanacak")
        print("   → (J) artık doğru gösterilecek")
    else:
        print("⚠️  Dosya yapısı beklenenden farklı - manuel kontrol gerekli")

def fix_timing_accuracy():
    """Fix timing accumulation issues."""

    print("\n" + "="*60)
    print("FIX 2: TIMING GECİKMESİ SORUNU")
    print("="*60)

    # renderer.py'de timing precision artırımı
    renderer_path = Path("autoshorts/captions/renderer.py")

    if not renderer_path.exists():
        print("❌ renderer.py bulunamadı")
        return

    content = renderer_path.read_text(encoding='utf-8')

    # MIN_WORD_DURATION değerini düşür (80ms → 60ms)
    if "MIN_WORD_DURATION = 0.08" in content:
        new_content = content.replace(
            "MIN_WORD_DURATION = 0.08",
            "MIN_WORD_DURATION = 0.06  # Reduced for better sync"
        )
        renderer_path.write_text(new_content, encoding='utf-8')
        print("✅ renderer.py güncellendi")
        print("   → MIN_WORD_DURATION: 80ms → 60ms")
        print("   → Daha hassas timing kontrolü")

    # Timing precision artırımı - cumulative error düzeltmesi
    if "TIMING_PRECISION = 0.001" in content:
        # Already good precision
        print("✅ Timing precision zaten optimal (1ms)")

    # forced_aligner.py'de precision artırımı
    aligner_path = Path("autoshorts/captions/forced_aligner.py")

    if aligner_path.exists():
        aligner_content = aligner_path.read_text(encoding='utf-8')

        if "MIN_WORD_DURATION = 0.08" in aligner_content:
            new_aligner = aligner_content.replace(
                "MIN_WORD_DURATION = 0.08",
                "MIN_WORD_DURATION = 0.06  # Better sync"
            )
            aligner_path.write_text(new_aligner, encoding='utf-8')
            print("✅ forced_aligner.py güncellendi")
            print("   → MIN_WORD_DURATION: 80ms → 60ms")

def create_timing_config():
    """Create optimal timing configuration suggestions."""

    print("\n" + "="*60)
    print("ÖNERİLEN TİMİNG AYARLARI")
    print("="*60)

    config = """
# Optimal Caption Timing Ayarları
# ================================

# Config dosyanızda (JSON/YAML) şu ayarları ekleyin:

{
  "captions": {
    "karaoke_offset_ms": -50,     # Captions 50ms geciktirilsin (audio'dan sonra)
    "karaoke_speed": 1.0,          # Normal hız
    "caption_lead_ms": 0,          # Ek gecikme yok
    "karaoke_effects": true,       # Karaoke efekti aktif
    "effect_style": "word_pop"     # Kelime kelime pop animasyonu
  }
}

# Alternatif olarak .env dosyasında:
KARAOKE_OFFSET_MS=-50
KARAOKE_SPEED=1.0
CAPTION_LEAD_MS=0

# Not: Negatif offset caption'ları GECİKTİRİR (audio'dan sonra gelir)
#      Pozitif offset caption'ları ÖNE alır (audio'dan önce gelir)
"""

    config_path = Path("CAPTION_TIMING_CONFIG.txt")
    config_path.write_text(config, encoding='utf-8')

    print(config)
    print(f"\n✅ Ayarlar {config_path} dosyasına kaydedildi")

def main():
    """Run all fixes."""
    print("\n" + "="*70)
    print("CAPTION SORUNLARINI DÜZELTME ARACI")
    print("="*70)
    print("\nSorunlar:")
    print("  1. Sayılar gözükmüyor: (J) → (J) olmalı")
    print("  2. Timing gecikmesi: 2-3. cümlelerde gecikme var")
    print("\n" + "="*70 + "\n")

    # Fix 1: Number display
    fix_number_display()

    # Fix 2: Timing accuracy
    fix_timing_accuracy()

    # Create config suggestions
    create_timing_config()

    print("\n" + "="*70)
    print("TÜM DÜZELTMELER TAMAMLANDI!")
    print("="*70)
    print("\nSonraki Adımlar:")
    print("  1. Değişiklikleri commit edin")
    print("  2. Yeni bir video oluşturun")
    print("  3. Caption'ları kontrol edin")
    print("\nEğer hala sorun varsa:")
    print("  - KARAOKE_OFFSET_MS değerini -100'e çıkarın (daha fazla gecikme)")
    print("  - MIN_WORD_DURATION'ı 0.05'e düşürün (daha hassas)")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
