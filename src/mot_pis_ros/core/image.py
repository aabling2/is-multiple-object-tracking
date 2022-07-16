import os


# Mapeia arquivos de imagens
def map_cam_image_files(src, ext='jpeg'):
    batch_files = {}
    N = 0
    for root, _, files in os.walk(src):
        if not files or root == '':
            continue

        batch_files[root] = sorted([x for x in files if os.path.splitext(x)[1] == f'.{ext}'])
        n = len(batch_files[root])
        N = n if n < N or N == 0 else N

    # Pós-processamento, organiza em grupos para cada amostragem conjunta
    join_files = []
    for i in range(N):
        join_files.append([os.path.join(os.path.basename(cam), filenames[i]) for cam, filenames in batch_files.items()])

    return join_files
