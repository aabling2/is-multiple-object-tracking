# mot-pis-ros

MOT (Multiple Object Tracking) integrado ao PIS (Programmable Intelligent Space) com ROS (Robot Operating System).


## Objetivos

- Criar framework de testes para múltiplas câmeras
    - Detector/classificador (Aiury)
    - Rastreio (MOT)
    - Análise e atribuição de objetos entre imagens (MultiCAM Re-ID)
    - Reconstrução do ponto central do objeto no espaço 3D
- Atacar componentes para melhorias
    - MOT
    - Re-ID multicam
    - Reconstrução 3D
- Desenvolver serviço(s) para retorno da posição/classe do objeto rastreado.
    - Acessar câmeras PIS
    - Adequar comunicação com PIS e/ou ROS
    - Adequar como serviço

## Metas

- [x] Desenvolver framework de testes, usando Yolov5, DeepSORT e alguns métodos básicos para reidentificação de objetos rastreados.
- [x] Baixar dataset do PIS e montar leitura dos frames simultâneos.
- [ ] Desenvolver método de rastreio que trate por padrão re-identificação em multiplas câmeras (checar se funcionamento se aplica nos casos de câmeras que não se sobrepõem no campo de visão).
- [ ] Desenvolver código que faça o consumo e fornecimento das mensagens contendo images, bounding boxes, labels, etc.
- [x] Desenvolver anotação de vídeo se não existir, para as câmeras do PIS.
    - [ ] Falta ajustar bboxes em cada frame, mais demorado.
- [ ] Desenvolver benchmark com métricas de desempenho que contemple avaliação de rastreio e re-identificação.
- [ ] Aplicar benchmark entre métodos de rastreio existentes (DeepSORT, FairMOT, ByteTrack, Tracktor++, StrongSORT, FastMOT).
- [ ] Melhorar metódo de rastreio para cumprir o objetivo de rastreio e re-identificação em multiplas câmeras.
- [ ] Desenvolver código para embarcar aplicação como serviço do PIS.
