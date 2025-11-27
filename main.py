import argparse
import json
import cv2
from grader import OMRGrader

def main():
    # Configurar argumentos da linha de comando
    parser = argparse.ArgumentParser(description="OMR Grader - Corretor Automático de Gabaritos")
    parser.add_argument("-i", "--image", required=True, help="Caminho para a imagem do gabarito")
    parser.add_argument("-d", "--debug", action="store_true", help="Ativar modo debug (salvar imagens intermediárias)")
    args = parser.parse_args()

    # Instanciar o corretor
    grader = OMRGrader(debug=args.debug)

    print(f"Processando imagem: {args.image}...")

    try:
        # Processar a imagem
        results = grader.process_image(args.image)
        
        # Exibir resultados no console
        print("\nResultados da Correção:")
        print(json.dumps(results, indent=4, sort_keys=True))
        
        # Salvar em arquivo JSON
        with open("results.json", "w") as f:
            json.dump(results, f, indent=4, sort_keys=True)
            
        print("\nResultados salvos em 'results.json'.")
        
        if args.debug:
            print("Imagens de debug salvas na pasta 'debug/'.")

    except Exception as e:
        print(f"\n[ERRO] Falha no processamento: {e}")

if __name__ == "__main__":
    main()
