from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import os
import cv2
import numpy as np
import json
from grader import OMRGrader

app = Flask(__name__)
CORS(app)  # Habilita CORS para todas as rotas

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/api/process', methods=['POST'])
def process_image():
    try:
        data = request.json
        
        if not data or 'image' not in data or 'gabarito' not in data:
            return jsonify({'success': False, 'error': 'Dados incompletos. Envie imagem e gabarito.'}), 400

        image_data = data['image']
        gabarito = data['gabarito']

        # Decodificar imagem base64
        try:
            # Remove o cabeçalho do base64 se existir (ex: "data:image/jpeg;base64,")
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            
            # Salvar imagem temporariamente
            temp_filename = os.path.join(UPLOAD_FOLDER, 'temp_upload.jpg')
            with open(temp_filename, 'wb') as f:
                f.write(image_bytes)
                
        except Exception as e:
            return jsonify({'success': False, 'error': f'Erro ao decodificar imagem: {str(e)}'}), 400

        # Processar imagem usando o OMRGrader
        try:
            grader = OMRGrader(debug=True) # Debug ativado por padrão para dev
            results = grader.process_image(temp_filename)
            
            # Comparar com o gabarito
            comparison = compare_results(results, gabarito)
            
            # Gerar imagem de resultado (opcional, por enquanto retornamos a original ou processada se o grader suportasse)
            # O grader salva imagens de debug, poderíamos retornar uma delas.
            # Por simplicidade, vamos retornar a imagem original em base64 por enquanto, 
            # ou idealmente a imagem marcada. O grader atual salva debug_5_final.jpg
            
            result_image = None
            debug_final_path = os.path.join('debug', 'debug_5_final.jpg')
            if os.path.exists(debug_final_path):
                with open(debug_final_path, "rb") as image_file:
                    result_image = "data:image/jpeg;base64," + base64.b64encode(image_file.read()).decode('utf-8')

            return jsonify({
                'success': True,
                'answers': results,
                'comparison': comparison,
                'result_image': result_image
            })

        except Exception as e:
            print(f"Erro no processamento: {e}")
            return jsonify({'success': False, 'error': f'Erro ao processar gabarito: {str(e)}'}), 500

    except Exception as e:
        return jsonify({'success': False, 'error': f'Erro interno do servidor: {str(e)}'}), 500

def compare_results(student_answers, gabarito):
    correct_count = 0
    total_questions = 0
    details = {}
    
    # O gabarito vem no formato: { questions: [ { id: 1, correctAnswer: 'A' }, ... ] }
    # Ou algo similar. Vamos verificar a estrutura no frontend ou assumir uma estrutura padrão.
    # Baseado no upload/page.tsx: gabarito.questions é um array.
    
    gabarito_map = {}
    if 'questions' in gabarito:
        for q in gabarito['questions']:
            # Assumindo que 'id' é o número da questão ou existe um campo 'number'
            # Vamos tentar usar o índice + 1 se não tiver número explícito, ou confiar na ordem
            q_num = q.get('number', q.get('id')) 
            gabarito_map[str(q_num)] = q.get('correctAnswer')
            
    # Se o mapa estiver vazio, tenta outra estratégia ou usa o índice
    if not gabarito_map and 'questions' in gabarito:
         for i, q in enumerate(gabarito['questions']):
            gabarito_map[str(i + 1)] = q.get('correctAnswer')

    # Calcular nota
    # student_answers é {'01': 'A', '02': 'B', ...}
    
    for q_num, student_ans in student_answers.items():
        # Remover zero à esquerda para comparar se necessário (ex: '01' -> '1')
        q_key = str(int(q_num)) 
        
        if q_key in gabarito_map:
            correct_ans = gabarito_map[q_key]
            is_correct = student_ans == correct_ans
            
            if is_correct:
                correct_count += 1
            
            details[q_key] = {
                'student': student_ans,
                'correct': correct_ans,
                'is_correct': is_correct
            }
            total_questions += 1
            
    score = (correct_count / total_questions * 100) if total_questions > 0 else 0
    
    return {
        'score': score,
        'correct': correct_count,
        'total': total_questions,
        'details': details
    }

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
