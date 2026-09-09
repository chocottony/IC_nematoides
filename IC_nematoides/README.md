# Classificação de Nematoides com ResNet-50

Pipeline de fine-tuning de ResNet-50 (pré-treinada no ImageNet) para
classificação de gênero de nematoides a partir de imagens de microscopia
(dataset I-Nema), como parte do projeto de Iniciação Científica
"Avaliação da Qualidade do Solo através da Análise de Nematoides,
Utilizando Algoritmo de Aprendizado de Máquina" (2026/2027).

## O que o script faz

`ResNet_multi_seed.py` executa o pipeline completo:

1. **Treina 3 rodadas independentes** (seeds 42, 123, 2024), cada uma do
   zero, para medir a variabilidade real do modelo em vez de confiar no
   resultado de uma única execução.
2. **Balanceamento dinâmico de classes** via `WeightedRandomSampler`
   (peso `1/count^0.5`), combinado com augmentation aplicado em tempo
   real (crop aleatório, flip, `ColorJitter`, erasing) — em vez de
   duplicar fisicamente imagens das classes raras no disco.
3. **Focal Loss** (Lin et al., 2017) no lugar de CrossEntropy simples,
   para concentrar o gradiente nas classes mais difíceis de
   classificar sem precisar forçar ainda mais o balanceamento de
   amostragem.
4. **Regularização** contra overfitting: dropout antes da camada final,
   label smoothing, weight decay maior, early stopping.
5. **Avaliação detalhada por rodada**: Top-1/Top-2 accuracy,
   precision/recall/F1 macro e ponderado, matriz de confusão, métricas
   por espécie.
6. **Agregação das 3 rodadas**: média ± desvio-padrão de cada métrica
   (global e por classe) — importante para reportar resultados
   estatisticamente honestos com um dataset pequeno e desbalanceado.
7. **Ensemble final**: combina os 3 modelos treinados por média das
   probabilidades softmax, geralmente superando o melhor modelo
   individual.

## Estrutura de dados esperada

```
I-Nema/
├── train/            <- usado por padrão (TRAIN_SUBDIR = "train")
│   ├── Acrobeles/
│   ├── Acrobeloides/
│   └── ...
├── train_balanced/    <- alternativa (balanceamento estático via
│                          augmentation pré-gerada); ver nota abaixo
└── val/
    ├── Acrobeles/
    └── ...
```

> **Nota sobre `train` vs. `train_balanced`**: testes mostraram que
> treinar sobre `train` (classes desbalanceadas) com o balanceamento
> dinâmico do sampler supera treinar sobre `train_balanced` (cópias
> estáticas pré-geradas), porque evita repetir os mesmos pixels
> exatos das classes raras a cada época. `train_balanced` fica
> disponível só para comparação/ablação.

## Configuração

Todos os hiperparâmetros ficam centralizados no dicionário `CONFIG`
no topo do script. Os principais:

| Parâmetro | Valor atual | Descrição |
|---|---|---|
| `sampler_power` | 0.5 | Expoente do peso do sampler (0 = sem balanceamento, 1 = balanceamento total) |
| `use_focal_loss` | True | Focal Loss em vez de CrossEntropy |
| `focal_gamma` | 2.0 | Intensidade do foco em exemplos difíceis |
| `dropout` | 0.3 | Antes da camada final |
| `freeze_mode` | 'partial' | Congela tudo exceto `layer4` e `fc` |
| `weight_decay` | 1e-3 | Regularização L2 |
| `label_smoothing` | 0.1 | Suavização dos rótulos |

Ajuste `I_NEMA_DIR` para o caminho local do dataset antes de rodar.

## Como rodar

```bash
poetry run python ResNet_multi_seed.py
```

## Saídas geradas (pasta `runs/`)

- `best_model_seed{42,123,2024}.pth` — checkpoints de cada rodada
- `run_seed{seed}_matriz.png` — matriz de confusão por rodada
- `ensemble_matriz.png` — matriz de confusão do ensemble
- `agregado_metricas_globais.csv` / `agregado_metricas_por_especie.csv`
  — média ± desvio-padrão das 3 rodadas
- `ensemble_metricas_globais.csv` / `ensemble_metricas_por_especie.csv`
  — resultado do ensemble

## Histórico de experimentos (dataset I-Nema, val N=552, 19 classes)

| Configuração | Top-1 | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| `train_balanced`, CrossEntropy (execução única) | 68,7% | 0,618 | 0,681 |
| `train` original, sampler power=1.0, CrossEntropy (média 3 seeds) | 61,4% | 0,536 | 0,611 |
| `train` original, sampler power=0.5, CrossEntropy (média 3 seeds) | 63,8% | 0,554 | 0,636 |
| `train` original, sampler power=0.5, Focal Loss (média 3 seeds) | 66,2% | 0,591 | 0,653 |
| **Ensemble (3 modelos, Focal Loss)** | **71,0%** | **0,663** | **0,705** |

**Limitação conhecida**: espécies da família Dorylaimida (Aporcelaimus,
Axonchium, Discolimus, Dorylaimus, Eudorylaimus, Mesodorylaimus)
apresentam confusão morfológica persistente entre si em todas as
configurações testadas — indicando um problema de classificação de
grão fino que não é resolvido apenas por ajuste de sampler/loss.
Próximo passo sugerido: classificação hierárquica (família → gênero)
e/ou coleta adicional de imagens reais para as classes com menor N
(Amplimerlinius, Dorylaimus).

## Requisitos

- Python 3.10+
- `torch`, `torchvision`, `matplotlib`, `tqdm`, `pillow`, `numpy`

## Observação sobre versionamento

Os arquivos `.pth` (checkpoints) são grandes (~100MB cada) — considere
adicionar `runs/*.pth` ao `.gitignore` em vez de versioná-los
diretamente no GitHub, a menos que esteja usando Git LFS.
