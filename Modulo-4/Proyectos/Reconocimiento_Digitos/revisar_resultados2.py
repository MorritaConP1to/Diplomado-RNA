import json

with open(r'D:\Diplomado-RNA\Modulo-4\Proyectos\Reconocimiento_Digitos\numero_completo_ejecutado.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

error_count = 0
for i, cell in enumerate(nb['cells']):
    cell_type = cell['cell_type']
    outputs = cell.get('outputs', [])
    
    for out in outputs:
        if out.get('output_type') == 'error':
            error_count += 1
            print('ERROR en celda {}: {}: {}'.format(i, out['ename'], out['evalue'][:300]))
            print()
        elif out.get('output_type') == 'stream':
            text = ''.join(out.get('text', []))
            if text.strip():
                lines = text.strip().split('\n')
                # Print first and last output lines
                first = lines[0][:100]
                last = lines[-1][:100]
                print('Celda {} stdout: {} ... {}'.format(i, first, last))

if error_count == 0:
    print('No se encontraron errores!')
