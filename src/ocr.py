import cv2
import easyocr

# Cache EasyOCR Reader instances to avoid reloading weights
_readers_cache = {}

def get_easyocr_reader(langs, gpu=False):
    key = (tuple(sorted(langs)), gpu)
    if key not in _readers_cache:
        _readers_cache[key] = easyocr.Reader(list(langs), gpu=gpu)
    return _readers_cache[key]

class PlateOCR:
    def __init__(self, langs=None, gpu=False):
        if langs is None:
            langs = ['en']
        self.langs = langs
        self.gpu = gpu
        self.reader = get_easyocr_reader(self.langs, self.gpu)

    def normalize(self, text):
        # Convert Hindi digits to English
        hindi_to_eng = {
            '०':'0','१':'1','२':'2','३':'3','४':'4',
            '५':'5','६':'6','७':'7','८':'8','९':'9'
        }
        for h, e in hindi_to_eng.items():
            text = text.replace(h, e)
        return self.correct_plate_pattern(text)

    def correct_plate_pattern(self, text):
        text = text.upper()
        
        # Strip leading IND if present
        if text.startswith("IND"):
            text = text[3:]
            
        text = "".join([c for c in text if c.isalnum()])
        n = len(text)
        
        if n < 5 or n > 12:
            return text
            
        char_to_num = {
            'O': '0', 'I': '1', 'Z': '2', 'S': '5', 
            'B': '8', 'A': '4', 'G': '6', 'T': '1', 'L': '4'
        }
        num_to_char = {
            '0': 'O', '1': 'I', '2': 'Z', '5': 'S', 
            '8': 'B', '4': 'A', '6': 'G'
        }
        
        char_list = list(text)
        
        # 1. First 2 characters must be letters (State code)
        for i in range(min(2, n)):
            if char_list[i].isdigit() and char_list[i] in num_to_char:
                char_list[i] = num_to_char[char_list[i]]
                
        # 2. Last 4 characters must be digits
        for i in range(max(0, n - 4), n):
            if i >= 2: # Keep first two as letters
                if char_list[i].isalpha() and char_list[i] in char_to_num:
                    char_list[i] = char_to_num[char_list[i]]
                    
        # 3. Middle part mapping (index 2 to n-5)
        middle_len = n - 6
        if middle_len > 0:
            if middle_len == 4: # e.g. RJ 14 CV 0002
                for i in [2, 3]:
                    if char_list[i].isalpha() and char_list[i] in char_to_num:
                        char_list[i] = char_to_num[char_list[i]]
                for i in [4, 5]:
                    if char_list[i].isdigit() and char_list[i] in num_to_char:
                        char_list[i] = num_to_char[char_list[i]]
            elif middle_len == 3: # e.g. RJ 14 C 0002
                for i in [2, 3]:
                    if char_list[i].isalpha() and char_list[i] in char_to_num:
                        char_list[i] = char_to_num[char_list[i]]
                if char_list[4].isdigit() and char_list[4] in num_to_char:
                    char_list[4] = num_to_char[char_list[4]]
            elif middle_len == 2: # e.g. RJ 1 C 0002
                if char_list[2].isalpha() and char_list[2] in char_to_num:
                    char_list[2] = char_to_num[char_list[2]]
                if char_list[3].isdigit() and char_list[3] in num_to_char:
                    char_list[3] = num_to_char[char_list[3]]
            elif middle_len == 1: # e.g. RJ 1 0002
                if char_list[2].isalpha() and char_list[2] in char_to_num:
                    char_list[2] = char_to_num[char_list[2]]
                    
        return "".join(char_list)

    def read_text(self, crop):
        if crop is None or crop.size == 0:
            return "", 0.0

        # Preprocessing variants
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        variants = []
        # 1. Original BGR crop
        variants.append(crop)
        # 2. Raw grayscale
        variants.append(gray)
        # 3. Otsu binarization
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(otsu)
        # 4. Inverse Otsu binarization
        _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        variants.append(otsu_inv)
        # 5. CLAHE Equalized Grayscale + Otsu binarization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_gray = clahe.apply(gray)
        _, clahe_otsu = cv2.threshold(clahe_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(clahe_otsu)

        best_text = ""
        best_conf = 0.0

        # Set allowlist if using English only (ignores punctuation and noise)
        allowlist = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" if self.langs == ['en'] else None

        for var in variants:
            # Resize small images to improve character resolution
            h_var, w_var = var.shape[:2]
            if h_var < 80 or w_var < 160:
                var = cv2.resize(var, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

            if allowlist:
                result = self.reader.readtext(var, allowlist=allowlist)
            else:
                result = self.reader.readtext(var)

            texts = []
            confs = []

            for res in result:
                texts.append(res[1])
                confs.append(res[2])

            if confs:
                avg_conf = sum(confs) / len(confs)
                if avg_conf > best_conf:
                    best_conf = avg_conf
                    best_text = " ".join(texts)

        return self.normalize(best_text), best_conf