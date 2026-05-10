import os
import json
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

# 1. AYARLARI YÜKLE (.env dosyasındaki keyleri okur)
load_dotenv()

# --- YAPILANDIRMA ---
TEST_MODU = True  # True: Kredi harcamaz, 'sonuc.json'dan okur. False: Roboflow'u çalıştırır.
GORSEL_YOLU = "test_fis.jpg"  # Fiş fotoğrafının adı

def fis_koordinatlarini_getir(gorsel_yolu):
    """Roboflow üzerinden veya yerel dosyadan koordinatları çeker."""
    if TEST_MODU:
        print("--- [BİLGİ] Test Modu Aktif: Koordinatlar 'sonuc.json' dosyasından okunuyor. ---")
        try:
            with open("sonuc.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print("HATA: 'sonuc.json' bulunamadı! Lütfen önce TEST_MODU = False yaparak bir kez çalıştır.")
            return None
    
    print("--- [BİLGİ] Canlı Mod Aktif: Roboflow API'ye bağlanılıyor... ---")
    client = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=os.getenv("ROBOFLOW_API_KEY")
    )
    
    # Roboflow'dan tahmin al (Senin model ID'n)
    result = client.infer(gorsel_yolu, model_id="receipt-rlnml/1")
    
    # Sonucu bir kez kaydet (Gelecekteki testler için)
    with open("sonuc.json", "w") as f:
        json.dump(result, f)
        
    return result

# 2. ANA AKIŞ
def main():
    # A. Roboflow Verisini Al
    rf_verisi = fis_koordinatlarini_getir(GORSEL_YOLU)
    if not rf_verisi: return

    # B. Gemini'yi Hazırla
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    
    # Senin ekran görüntüsünde gördüğümüz o yeni model
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
    except:
        # Eğer sistem henüz 2.5'i API üzerinden tanımıyorsa 1.5'e düşer (B planı)
        model = genai.GenerativeModel('gemini-1.5-flash')

    # C. Görseli Hazırla
    img = Image.open(GORSEL_YOLU)

    # D. Gemini'ye Komut (Prompt) Gönder
    prompt = f"""
    Sana bir alışveriş fişi fotoğrafı ve Roboflow tarafından tespit edilen şu koordinat verilerini veriyorum:
    {json.dumps(rf_verisi)}

    Lütfen bu bilgileri kullanarak:
    1. Mağaza/Market adını tespit et.
    2. Fişteki ürünleri ve yanlarındaki fiyatları liste halinde yaz.
    3. Toplam tutarı ve KDV bilgisini (varsa) belirt.
    4. Fişin tarih ve saatini ekle.
    
    Yanıtı tamamen Türkçe ve düzenli bir formatta ver.
    """

    print("--- [BİLGİ] Gemini 2.5 Flash analiz yapıyor... ---")
    response = model.generate_content([prompt, img])

    # E. Sonucu Yazdır
    print("\n" + "="*50)
    print("🤖 YAPAY ZEKA FİŞ ANALİZ SONUCU")
    print("="*50)
    print(response.text)
    print("="*50)

if __name__ == "__main__":
    main()