import json

with open(r'D:\Diplomado-RNA\Modulo-4\Proyectos\Reconocimiento_Digitos\numero_completo_ejecutado.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and cell.get('outputs'):
        for out in cell['outputs']:
            if out.get('output_type') == 'error':
                print('ERROR en celda {}: {}'.format(i, out['ename']))
                print('  ' + out['evalue'][:200])
            elif out.get('output_type') == 'stream' and out.get('name') == 'stdout':
                text = ''.join(out['text'])
                if 'Exactitud' in text or 'Accuracy' in text or 'Mejor accuracy' in text:
                    print('Celda {}: {}'.format(i, text.strip()))
                if '=' * 20 in text:
                    print('FOOTER celda {}:'.format(i))
                    print(text[:500])
