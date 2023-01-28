# is-multiple-object-tracking

MOT (Multiple Object Tracking) integrado ao PIS/IS (Programmable Intelligent Space).

## MOT-BYTETrack Service

Serviço de rastreio de múltiplos objetos em múltiplas câmeras, integrado ao sistema de comunicação via espaço inteligete. A aplicação foi baseada na metodologia de rastreio [BYTETrack](https://github.com/ifzhang/ByteTrack), para o funcionamento correto, como o artigo menciona, é necessário utilizar TODAS detecções feitas pelo detector (incluindo de score baixo).

Seguindo o conceito e formato de mensagens conforme a biblioteca is-msgs, esta aplicação tem suporte a fornecer os dados de anotação dos objetos rastreados e o frame renderizado com a representação das mesmas anotações.

### Funcionamento

* Principal: Anotações de algum detector --> MOT-BYTETrack (lógica de rastreio dos objetos) --> Anotações de rastreio
* Opcional: Frame de alguma câmera --> MOT-BYTETrack (renderiza detecções no frame) --> Frame renderizado

### Instalar dependências

* Requisitos: Python3 (3.6-3.8) e instalador de pacotes Pip.

```bash
cd src
pip3 install Cython numpy  # presente no requirements mas é requisitado previamente
pip3 install -r requirements.txt
```

### Conexão com broker

Iniciar conexão com broker que será o meio de troca de mensagens entre detector, rastreio e saída dos dados.

```bash
# Broker docker simulado (is-wire)
sudo docker run -d --rm -p 5672:5672 -p 15672:15672 rabbitmq:3.7.6-management
```

ou

```bash
# Broker espaço inteligente (VPN), instruções internas do laboratório para obter arquivo de configuração
sudo openvpn --config client.conf
```

### Testar algoritmo principal de rastreio de múltiplos objetos em múltiplas câmeras

```bash
# Roda script principal de rastreio (apenas consumo e publicação de anotações), --help para parâmetros extras
# configurações podem ser editadas pelo arquivo options.json em ./etc
python -m is_mot_bytetrack.main
```

### Ferramentas para simular fornecimento e visualização de vídeo e detecções (OPCIONAL)

#### Gerar stream de vídeo e detecções randômicas para alimentar sistema

```bash
# Simula fornecimento de imagem e anotação atráves de vídeo local e detecções artificiais
python -m is_mot_tools.simul_detector --source ~/Videos/video1.mp4,~/Videos/video2.mp4
```

#### Visualizar vídeo dos resultados do rastreio

```bash
# Simula obtenção dos dados do rastreio e exibe vídeo na janela com a representação dos objetos
# bbox das anotações recebidas em vermelho sobre a imagem renderizada
python -m is_mot_tools.simul_viewer
```
