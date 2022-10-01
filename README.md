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
- [x] Desenvolver método de rastreio que trate por padrão re-identificação em multiplas câmeras.
- [x] Desenvolver código para anotação de vídeo, para as câmeras do PIS.
- [ ] Desenvolver código que faça o consumo e fornecimento das mensagens contendo images, bounding boxes, labels, etc.
- [ ] Desenvolver código para embarcar aplicação como serviço do PIS.


## Ambiente de testes

Iniciar conexão com broker que será o meio de troca de mensagens entre detector, rastreio e saída dos dados.

### Conexão com broker

```
# Broker docker simulado
sudo docker run -d --rm -p 5672:5672 -p 15672:15672 rabbitmq:3.7.6-management
```

```
# Broker espaço inteligente (VPN), instruções internas do laboratório para obter arquivo de configuração
sudo openvpn /etc/openvpn/client/client.conf
```