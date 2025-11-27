import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from imutils import contours as imutils_contours
import os

class OMRGrader:
    """
    Classe responsável pelo processamento e correção de gabaritos OMR.
    Algoritmo de Alto Contraste para superfícies ruidosas (Granito).
    """

    def __init__(self, debug=True, debug_folder="debug"):
        self.debug = debug
        self.debug_folder = debug_folder
        if self.debug and not os.path.exists(self.debug_folder):
            os.makedirs(self.debug_folder)

    def process_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

        # ---------------------------------------------------------
        # ETAPA 1: LOCALIZAÇÃO DA FOLHA E ÂNCORAS (ANTI-GRANITO)
        # ---------------------------------------------------------
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Filtro Bilateral: Excelente para manter bordas dos quadrados mas alisar o granito
        blurred = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # 2. DETECÇÃO DE BORDAS INTELIGENTE (Canny)
        # Em vez de threshold de cor, usamos bordas. O quadrado tem bordas retas perfeitas.
        # O granito tem bordas caóticas.
        edged = cv2.Canny(blurred, 30, 200)
        
        if self.debug:
            cv2.imwrite(f"{self.debug_folder}/debug_1_edges.jpg", edged)
            
        # 3. Encontrar contornos na imagem de bordas
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        anchors = []
        # Ordenar por área (do maior para o menor) para pegar a folha ou os quadrados
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        for c in cnts:
            area = cv2.contourArea(c)
            # Filtro de tamanho: Quadrados são pequenos, mas não minúsculos
            if area < 200: continue 
            
            peri = cv2.arcLength(c, True)
            # Aproximação Poligonal
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            
            # Se tiver 4 vértices, é candidato a âncora
            if len(approx) == 4:
                (x, y, w, h) = cv2.boundingRect(approx)
                ar = w / float(h)
                
                # Aspect Ratio: Tem que ser quadrado (0.7 a 1.3)
                if 0.7 <= ar <= 1.3:
                    # Filtro de Solidez (Crucial):
                    # O Canny gera contornos "ocos". Vamos verificar se, na imagem original,
                    # a área dentro desse quadrado é ESCURA.
                    
                    # Cria uma máscara para ver o que tem dentro desse contorno
                    mask = np.zeros(gray.shape, dtype="uint8")
                    cv2.drawContours(mask, [c], -1, 255, -1)
                    
                    # Calcula a média de cor dentro do quadrado na imagem original
                    mean_val = cv2.mean(gray, mask=mask)[0]
                    
                    # Se a média for escura (< 100), é tinta preta (âncora).
                    # Se for clara (> 150), é papel ou granito claro.
                    if mean_val < 120:
                        anchors.append(approx)

            if len(anchors) == 4: break

        # SEGUNDA TENTATIVA: Se o Canny falhar (luz ruim), tenta Threshold Adaptativo
        if len(anchors) != 4:
            print("[INFO] Método Canny falhou nas âncoras. Tentando Threshold Adaptativo...")
            anchors = []
            # BlockSize gigante (51) para ignorar textura fina do granito
            thresh = cv2.adaptiveThreshold(blurred, 255,
                cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 15) # C=15 corta cinza
            
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            
            for c in cnts:
                area = cv2.contourArea(c)
                if area < 200: continue
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.04 * peri, True)
                if len(approx) == 4:
                    (x, y, w, h) = cv2.boundingRect(approx)
                    ar = w / float(h)
                    if 0.7 <= ar <= 1.3:
                        # Teste de convexidade para evitar manchas irregulares
                        if cv2.isContourConvex(approx):
                            anchors.append(approx)
                if len(anchors) == 4: break

        if len(anchors) != 4:
            if self.debug:
                debug_fail = image.copy()
                cv2.drawContours(debug_fail, cnts[:50], -1, (0, 0, 255), 2)
                cv2.imwrite(f"{self.debug_folder}/debug_falha_ancoras.jpg", debug_fail)
            raise Exception(f"Falha crítica nas âncoras. Encontrados: {len(anchors)} (Esperado: 4).")

        # Ordenar e Perspectiva (Código Padrão)
        src_pts = []
        for c in anchors:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                src_pts.append([cX, cY])
            else:
                x, y, w, h = cv2.boundingRect(c)
                src_pts.append([x + w//2, y + h//2])
        
        src_pts = np.array(src_pts, dtype="float32")
        src_pts = self.order_points(src_pts)
        warped, _ = self.four_point_transform_with_padding(image, src_pts, padding=20)
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        img_h, img_w = warped.shape[:2]

        # ---------------------------------------------------------
        # ETAPA 2: CAMADA 1 - O "MAPA" (Estrutura do Gabarito)
        # ---------------------------------------------------------
        blurred_grid = cv2.GaussianBlur(warped_gray, (3, 3), 0)
        thresh_grid = cv2.adaptiveThreshold(blurred_grid, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 2)
        
        cnts_grid = cv2.findContours(thresh_grid.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_grid = imutils.grab_contours(cnts_grid)
        
        all_bubbles = []
        margin = 15
        
        for c in cnts_grid:
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            if x > margin and y > margin and (x+w) < (img_w - margin) and (y+h) < (img_h - margin):
                if w >= 16 and h >= 16 and 0.7 <= ar <= 1.3:
                    if w < img_w * 0.06:
                        all_bubbles.append(c)

        if not all_bubbles:
            raise Exception("Erro: Nenhuma bolha de gabarito encontrada na imagem corrigida.")

        all_bubbles = imutils_contours.sort_contours(all_bubbles, method="left-to-right")[0]
        columns = []
        current_col = [all_bubbles[0]]
        gap_threshold_x = img_w * 0.10 

        for i in range(1, len(all_bubbles)):
            curr_x = cv2.boundingRect(all_bubbles[i])[0]
            prev_x = cv2.boundingRect(all_bubbles[i-1])[0]
            if curr_x - prev_x > gap_threshold_x:
                columns.append(current_col)
                current_col = []
            current_col.append(all_bubbles[i])
        columns.append(current_col)
        
        columns = [c for c in columns if len(c) >= 5]
        columns.sort(key=lambda col: cv2.boundingRect(col[0])[0])

        if self.debug:
            debug_map = warped.copy()
            for i, col in enumerate(columns):
                for c in col:
                    cv2.drawContours(debug_map, [c], -1, (255, 0, 0), 2)
            cv2.imwrite(f"{self.debug_folder}/debug_mapa_estrutural.jpg", debug_map)

        # ---------------------------------------------------------
        # ETAPA 3: CAMADA 2 - O "RAIO-X" (Marcas)
        # ---------------------------------------------------------
        blurred_marks = cv2.GaussianBlur(warped_gray, (11, 11), 0)
        thresh_marks = cv2.adaptiveThreshold(blurred_marks, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 15)
        
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh_marks = cv2.morphologyEx(thresh_marks, cv2.MORPH_ERODE, kernel_clean, iterations=1)
        thresh_marks = cv2.morphologyEx(thresh_marks, cv2.MORPH_DILATE, kernel_clean, iterations=1)

        if self.debug:
            cv2.imwrite(f"{self.debug_folder}/debug_raio_x.jpg", thresh_marks)

        # ---------------------------------------------------------
        # ETAPA 4: CRUZAMENTO (SCORING)
        # ---------------------------------------------------------
        results = {}
        global_question_index = 1
        answer_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
        final_debug_img = warped.copy()

        for col_cnts in columns:
            col_cnts = imutils_contours.sort_contours(col_cnts, method="top-to-bottom")[0]
            
            rows = []
            if len(col_cnts) > 0:
                current_row = [col_cnts[0]]
                prev_y = cv2.boundingRect(col_cnts[0])[1]
                for i in range(1, len(col_cnts)):
                    y = cv2.boundingRect(col_cnts[i])[1]
                    if abs(y - prev_y) < 20: 
                        current_row.append(col_cnts[i])
                    else:
                        rows.append(current_row)
                        current_row = [col_cnts[i]]
                    prev_y = y
                rows.append(current_row)

            for row in rows:
                row = imutils_contours.sort_contours(row, method="left-to-right")[0]
                question_num = f"{global_question_index:02d}"
                
                detected_bubbles = []
                for i, bubble_cnt in enumerate(row):
                    mask = np.zeros(thresh_marks.shape, dtype="uint8")
                    cv2.drawContours(mask, [bubble_cnt], -1, 255, -1)
                    mask = cv2.bitwise_and(thresh_marks, thresh_marks, mask=mask)
                    total_pixels = cv2.countNonZero(mask)
                    detected_bubbles.append((total_pixels, i, bubble_cnt))

                detected_bubbles.sort(key=lambda x: x[0], reverse=True)
                winner_count, winner_idx, winner_cnt = detected_bubbles[0]
                runner_up_count = detected_bubbles[1][0] if len(detected_bubbles) > 1 else 0
                
                if winner_count < 50:
                    answer = "NULA"
                    color = (0, 0, 255) 
                    draw_cnt = row[0] 
                else:
                    if runner_up_count > 0 and (winner_count / runner_up_count < 1.5):
                        answer = "DUVIDA"
                        color = (0, 255, 255)
                        draw_cnt = winner_cnt
                    else:
                        if winner_idx < 5:
                            answer = answer_map[winner_idx]
                        else:
                            answer = "?"
                        color = (0, 255, 0)
                        draw_cnt = winner_cnt

                if answer != "NULA":
                    cv2.drawContours(final_debug_img, [draw_cnt], -1, color, 3)
                    cv2.putText(final_debug_img, f"{question_num}:{answer}", 
                                (cv2.boundingRect(draw_cnt)[0]-10, cv2.boundingRect(draw_cnt)[1]-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                results[question_num] = answer
                global_question_index += 1

        if self.debug:
            cv2.imwrite(f"{self.debug_folder}/debug_5_final.jpg", final_debug_img)

        return results

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def four_point_transform_with_padding(self, image, pts, padding=0):
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([
            [0 + padding, 0 + padding],
            [maxWidth - 1 + padding, 0 + padding],
            [maxWidth - 1 + padding, maxHeight - 1 + padding],
            [0 + padding, maxHeight - 1 + padding]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth + 2*padding, maxHeight + 2*padding))
        return warped, M 