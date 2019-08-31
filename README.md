# Inception-Resnet-V2

Inception-v4
O Inception-v4, desenvolvido a partir do GoogLeNet / Inception-v1, possui uma arquitetura simplificada mais uniforme e mais módulos de criação do que o Inception-v3.
Esta é uma variante inicial pura, sem quaisquer conexões residuais. Ele pode ser treinado sem particionar as réplicas, com otimização de memória para retropropagação.
Podemos ver que as técnicas de Inception-v1 a Inception-v3 são usadas.

Residual Inception Blocks
Cada bloco de início é seguido por uma camada de expansão de filtro (1 × 1 convolução sem ativação) que é usado para escalar a dimensionalidade do banco de filtros antes da adição para coincidir com a profundidade da entrada.
No caso do Inception-ResNet, a normalização em lote é usada apenas no topo das camadas tradicionais, mas não no topo das somatórias.

Scaling of Residuals

Segundo os autores, se o número de filtros ultrapassou 1000, as variantes residuais começaram a exibir instabilidades e a rede acabou “morrendo” no início do treinamento, o que significa que a última camada antes do pool médio começou a produzir apenas zeros depois de alguns dezenas de milhares de iterações. 
Isso não poderia ser evitado, nem diminuindo a taxa de aprendizado, nem adicionando uma normalização extra a essa camada.


implementations of the Inception-v4, Inception - Resnet-v1 and v2 Architectures in Keras using the Functional API. The paper on these architectures is available at "Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning".

The models are plotted and shown in the architecture sub folder. Due to lack of suitable training data (ILSVR 2015 dataset) and limited GPU processing power, the weights are not provided.
 ##############

The python script 'inception_resnet_v2.py' contains the methods necessary to create the Inception ResNet v2 network. It is to be noted that scaling of the residuals is turned ON by default.

There are a few differences in the v2 network from the original paper:

[1] In the B blocks: 'ir_conv' nb of filters is given as 1154 in the paper, however input size is 1152.
This causes inconsistencies in the merge-sum mode, therefore the 'ir_conv' filter size is reduced to 1152 to match input size.

[2] In the C blocks: 'ir_conv' nb of filter is given as 2048 in the paper, however input size is 2144.

This causes inconsistencies in the merge-sum mode, therefore the 'ir_conv' filter size is increased to 2144 to match input size.

#################
Usage:

from inception_resnet_v2 import create_inception_resnet_v2

model = create_inception_resnet_v2(scale=True)


