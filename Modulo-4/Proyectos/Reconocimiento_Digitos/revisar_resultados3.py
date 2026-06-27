import json

with open(r'D:\Diplomado-RNA\Modulo-4\Proyectos\Reconocimiento_Digitos\numero_completo_ejecutado.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print('Total cells: {}'.format(len(nb['cells'])))
for i, cell in enumerate(nb['cells']):
    outputs = cell.get('outputs', [])
    if outputs:
        print('Cell {} has {} outputs:'.format(i, len(outputs)))
        for out in outputs:
            otype = out.get('output_type', 'unknown')
            if otype == 'stream':
                text = ''.join(out.get('text', []))
                print('  stream ({}): {}'.format(out.get('name', '?'), text[:200]))
            elif otype == 'error':
                print('  ERROR: {}'.format(out['evalue'][:200]))
            elif otype == 'display_data':
                print('  display_data (plot/rich output)')
            elif otype == 'execute_result':
                text = ''.join(out['data'].get('text/plain', []))
                print('  result: {}'.format(text[:200]))

print()
print('No more outputs beyond this point.')
