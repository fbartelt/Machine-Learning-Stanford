# Machine Learning

---

[TOC]

# Supervised Learning

Já se tem um banco de dados que mostra dados e os outputs esperados pelo computador.

## Modelo

| $m$         | número de exemplos de treino                      |
| ----------- | ------------------------------------------------- |
| $x$         | **variáveis de entrada (features)**               |
| $y$         | **variáveis de saída ou variáveis alvo (target)** |
| $(x,y)$     | **UM exemplo de treino**                          |
| $(x^i,y^i)$ | **i-ésimo exemplo de treino**                     |

O algoritmo gera, a partir do training set, uma hipótese $h$, função que dada uma nova entrada, gera uma resposta baseada no que aprendeu ($h:X\rightarrow Y$). No exemplo de regressão, prediz o preço de uma casa de um tamanho não listado anteriormente. 

![image-20201122171240946](./imgs/image-20201122171240946.png)

## Regression

Estipular valores contínuos intermediários e futuros a partir de um banco de dados (Training Set). Exemplo: estimar preço de venda de casa

### Regressão linear de uma variável

$$
h_\theta(x) = \theta_0+\theta_1x
$$
$\theta_i$ são os parâmetros do modelo. Tenta-se defini-los de forma que a saída de $h_\theta$ seja a mais próxima possível dos valores do training set $(x,y)$. A definição dos parâmetros pode ser feita pelo erro quadrático médio (MSE):

$$
\min_{\theta_0,\theta_1}\frac{1}{2m}\sum_{i=1}^m(h_\theta(x^i)-y^i)^2 = \min_{\theta_0,\theta_1}J(\theta_0,\theta_1),
$$

Onde $J(\theta_0,\theta_1)$ é chamada função de custo

A média é dividida por $2$ por conveniência, de forma que a derivada no método do gradiente tenha o $2$ anulado.

### Regressão linear multivariável

| $n$     | número de features                             |
| ------- | ---------------------------------------------- |
| $x^i$   | inputs (features) do i-ésimo exemplo de treino |
| $x_j^i$ | valor da feature $j$ do i-ésimo exemplo        |

$$
h_\theta(x) = \theta_0+\theta_1x_1+\theta_2x_2+\dots+\theta_nx_n
$$

Tomando $x_0^i=1\;\forall \;i\in\{1,2,\dots,m\}$, tem-se uma notação mais simples por:
$$
x= \begin{bmatrix}x_0\\x_1\\\vdots\\x_n\end{bmatrix}\in\R^{n+1},
\quad \theta= \begin{bmatrix}\theta_0\\\theta_1\\\vdots\\\theta_n\end{bmatrix}\in\R^{n+1}
$$

$$
h_\theta(x) = \theta^T\cdot x
$$

## Método do Gradiente

Algoritmo para minimização. Para o caso da função de custo $J(\theta_0,\theta_1)$, toma-se estimativas iniciais de $\theta_0,\theta_1$ e se encontra o mínimo local iterativamente. Sem condições iniciais, toma-se ambos como $0$ :
$$
\large\theta_j:= \theta_j-\alpha\frac{\partial J(\theta_0,\theta_1)}{\partial\theta_j}
$$

$\alpha$ - escalar chamado de *Learning Rate*, tipo uma escala pro passo da iteração

Deve-se fazer a operação simultaneamente em cada $\theta_n$, de forma a não utilizar um valor atualizado para atualizar o próximo parâmetro.

### Aplicação na função de custo da regressão

$$
\large\frac{\partial J(\theta_0,\theta_1)}{\partial\theta_j} = \frac{\partial}{\partial\theta_j}\frac{1}{2m}\sum_{i=1}^m(h_\theta(x^i)-y^i)^2 = \frac{\partial}{\partial\theta_j}\frac{1}{2m}\sum_{i=1}^m(\theta_0+\theta_1x^i-y^i)^2
$$

Tem-se:
$$
\begin{cases}
\large{
\theta_0\implies\frac{\partial}{\partial\theta_0}=\frac{1}{m}\sum_{i=1}^m(h_\theta(x^i)-y^i)\\
\theta_1\implies\frac{\partial}{\partial\theta_1}=\frac{1}{m}\sum_{i=1}^m(h_\theta(x^i)-y^i)x^i_1
}
\end{cases}
$$
Portanto:
$$
\begin{align}
\theta_0&:= \theta_0-\alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^i)-y^i)\\
\theta_1&:= \theta_1-\alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^i)-y^i)x^i_1
\end{align}
$$

### Aplicação multivariável

$$
J(\theta_0,\theta_1,\dots,\theta_n)=J(\theta)=\frac{1}{2m}\sum_{i=1}^m(h_\theta(x^i)-y^i)^2\\
\theta_j:= \theta_j-\alpha\frac{\partial J(\theta)}{\partial\theta_j}
$$

Tratando tudo como vetores.
$$
\theta_j:=\theta_j-\alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^i)-y^i)x^i_j
$$

### Feature Scaling

Facilitar a convergência do método, fazendo com que as features tenham limites similares. Ex: se $x_1$ vai de 0-2000 e $x_2$ de 1-5, então dividir cada valor pelo limite máximo de cada, torna a convergência melhor, $\frac{x_1^i}{2000}$ e $\frac{x_2^i}{5}$

![image-20201123163636444](./imgs/image-20201123163636444.png)

A ideia é que se tenha $-1\le x_j\le1$ aproximadamente, pode-se ter valores próximos. De forma genérica, é aceitável $-3\le x_j\le3$ ou $-\frac13\le x_j\le\frac13$. 

### Mean Normalization

Troca o $x_j$ por $x_j-\mu_j$ para tornar a média próxima de 0 (não aplicando a $x_0$ e depois adiciona essa coluna): $x_j=\large{\frac{x_j-\mu_j}{\max x_j-\min x_j}}$, ou só max. Pode-se dividir ainda pelo desvio padrão, de forma a melhorar a aproximação. Mesmo processo anterior, porém subtraindo a média $\mu_j$ de cada feature. Torna um limite próximo de $-0.5\le x_j\le0.5$.

Para processos iterativos, é importante armazenar as médias e desvio padrões ($\mu$ e o divisor, no caso) para que previsões de valores futuros tenha a mesma normalização que as demais *features*.

### Debugging

Testar se o método do gradiente está funcionando corretamente: plotar o $\min J(\theta) \times$ iterações e ver se forma um decrescimento exponencial

### Escolha de  $\large{\mathbb{\alpha}}$

Testar uma escala de valores possíveis para o *Learning rate* $\alpha$, $0.001, 0.003, 0.01,\dots,1,\dots$ e escolher o maior que converge $J$ ou o menor anterior a este.

## Polynomial regression

Pode-se definir novas *features* com base nas existentes. Ex: se tem largura e comprimento, pode-se definir uma *feature* área para a estimação de parâmetros para venda de casa. Assim, é possível obter modelos melhores ao se definir novas *fts* e aplicar uma regressão correspondente:
$$
h(\theta) = \theta_0+\theta_1x+\theta_2x^2+\theta_3x^3\\
h(\theta) = \theta_0+\theta_1x+\theta_2\sqrt{x}
$$
Ao se analisar o gráfico do banco de dados, percebe-se que essas regressões polinomiais são similares ao gráfico de dados.

## Normal Equation

Outro método de minimizar a função de custo. Método direto, não necessitando várias iterações e definição do parâmetro $\alpha$

![image-20201201155736108](./imgs/image-20201201155736108.png)

Para minimizar uma função $J(\theta_0, \theta_1,\dots,\theta_n)$:
$$
\theta = (X^TX)^{-1}X^Ty
$$
$X$ - *design matrix*, formada pelos vetores de feature como colunas, ou exemplos de treino como linhas (com adição de $x_0$).

Se $X$ não é invertível, pode-se utilizar a pseudo-inversa.

Com o método de função normal, torna-se desnecessário fazer *feature scaling*.

|     Método do gradiente     |        Equação normal         |
| :-------------------------: | :---------------------------: |
|  Precisa escolher $\alpha$  | Não precisa escolher $\alpha$ |
| Precisa de várias iterações |      Não precisa iterar       |
|   Bom quando $n$ é grande   |    é ruim para $n$ grande     |
|          $O(kn^2)$          | $(X^TX)^{-1} \implies O(n^3)$ |

$n = 10000$ já pode começar a pensar em gradiente. 

## Classification

Estipular um resultado discreto (sim ou não), intermediário ou futuro, a partir de um banco de dados (estimar se cancer é benigno ou não; email: spam ou nao).

Usar regressão linear não é bom:

![image-20201213114825541](./imgs/image-20201213114825541.png)

> A classificação de forma binária faz com que um exemplo de treino possa afetar muito a predição, tornando-a ruim.

### Logistic Regression

Pretende-se ter uma hipótese de forma que $0\le h_\theta(x)\le1$. Então $h_\theta(x)=g(\theta^Tx)$
$$
\begin{align}
g(z) &= \frac{1}{1+e^{-z}}
\\\implies h_\theta(x) &= \frac{1}{1+e^{-\theta^Tx}}
\end{align}
$$
$g(z)$ - Sigmoid function (logistic function)

![image-20201213120136986](./imgs/image-20201213120136986.png)

$h_\theta(x)$ passa a estimar a probabilidade de um novo input ser $1$, dado $x$ parametrizado por $\theta$:

$h_\theta(x)=P(y=1\;|\;x\,;\,\theta) = 1-P(y=0\;|\;x\,;\,\theta)$ 

### Decision Boundary

Se $y=1$ se $h_\theta(x)\ge0.5$ e $y=0$ se $h_\theta(x)<0.5$ . Então $y=1$ se $g(\theta^Tx)\ge0.5\implies\theta^Tx\ge0$. 

Supondo $\theta=(-3 \,\ 1 \,\ 1)$, então $y=1 $ se $x_1+x_2\ge3$, e $y=0$ caso contrário. Isso forma uma reta chamada Decision Boundary em $x_1+x_2=3$, onde $h_\theta(x)=0.5$:

![image-20201213130613230](./imgs/image-20201213130613230.png)

> A Decision Boundary é definida pelo parâmetro $\theta$ e não pelo *training set*, pelo menos não diretamente. É definida pela hipótese.

### Non Linear Decision Boundaries

Pode-se utilizar uma função polinomial para aproximar melhor, de forma que

![image-20201213133151428](./imgs/image-20201213133151428.png)

### Cost Function

Primeiro, define-se uma função para substituir o erro quadrático médio, que não funciona para *Logistic Regression*, já que a função não seria convexa:
$$
Cost(h_\theta(x),y)=\begin{cases}
-\log(h_\theta(x)),&\text{se}\quad y=1\\ -\log(1-h_\theta(x)),&\text{se}\quad y=0
\end{cases}
$$
Define-se então a função de custo, agora convexa, para *Logistic Regression*:
$$
J(\theta)=\frac1m\sum_{i=1}^m Cost(h_\theta(x^i),y^i)
$$
![image-20201213163641290](./imgs/image-20201213163641290.png)

![image-20201213164443548](./imgs/image-20201213164443548.png)

Ainda, para o caso binário $y \in \{0,1\}$, pode-se escrever $J(\theta)$ da seguinte forma:
$$
J(\theta)=-\frac1m\sum_{i=1}^m y^i\log(h_\theta(x^i))+(1-y^i)\log(1-h_\theta(x^i))
$$
Tomando o mesmo procedimento com o método gradiente, tomando derivada etc. encontra-se a mesma expressão para a minimização do $J(\theta)$ por meio de $\theta$:
$$
\theta_j:=\theta_j-\alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^i)-y^i)x^i_j
$$
Vantagens de métodos de otimização quasi-newton e o gradiente conjugado: não precisa escolher $\alpha$ mabualmente e converge mais rápido.

### Multiclass Classification

Ex: automaticamente separar email em Trabalho, Hobby, Amigos. Diagnóstico médico, previsão do tempo, etc

O problema deixa de ser binário. Utiliza-se o método **One vs All **(ou *One vs rest*):

Pra cada classificação, existirá um problema de classificação binário. Então se as classificações são A, B e C. Tomar-se-á uma classificação binária de ser A ou não, ser B ou não ...

![image-20201213182326054](./imgs/onevall.png)

Então, ter-se-á $n$ hipóteses para as $n$ classificações do problema, de forma a computar a probabilidade de $y=n$. Assim, para encontrar a classe $z$ de um novo *input* $x$, toma-se o $n$ que maximiza a hipótese:
$$
z=\max_i h^i_\theta(x)
$$

## Regularization

O problema do *Overfitting*: toma-se tantas *features* que a regressão acaba virando interpolação, impedindo que valores futuros sejam previstos, porém obtendo uma estimativa excelente de valores intermediários. Obviamente que a função de custo terá valor nulo para o *training set*  e "previsão" de valores dentro do *training set* será perfeita, porém não o será para outros *inputs*

![image-20201213190230843](./imgs/overfit.png)

![](./imgs/2020-12-13_19-12.png)

Para corrigir/evitar : 

1. Reduzir número de *features*
   - Manualmente selecionar *features* que realmente são necessárias. 
   - Usar *Model Selection Algorithm*
2. Regularization
   - Manter todas as *features*, mas diminuir a magnitude/valor dos parâmetros $\theta_j$
   - Funciona bem quando se tem muitas *features*, quando cada uma contribui um pouco para prever $y$

A ideia de regularização é ter valores pequenos para $\theta_i$, tornando a hipótese mais simples e diminuindo a chance de *overfitting*. Para forçar isso, no caso de regressão linear, basta fazer uma penalização de todos os $\theta$, com exceção de $\theta_0$ :
$$
J(\theta)=\frac{1}{2m}\left(\sum_{i=1}^m(h_\theta(x^i)-y^i)^2 +\lambda\sum_{j=1}^n\theta_j^2\right)
$$
$\lambda$ - *regularization parameter* : controla um trade-off entre aproximar bem os *training sets* (primeiro somatorio) e manter os parâmetros pequenos (feito pelo segundo somatório).

A escolha do parâmetro de regularização deve ser cuidadosa, já que um valor muito grande pode ocasionar em *underfitting*

### Na regressão linear

Para o método gradiente, toma-se a modificação:
$$
\begin{cases}
\theta_0&:= \theta_0-\alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^i)-y^i)\\
\theta_j&:= \theta_j-\alpha\Big(\frac{1}{m}\sum_{i=1}^m(h_\theta(x^i)-y^i)x^i_j+\frac\lambda m \theta_j\Big)
\end{cases}
$$
Pode-se escrever de forma reduzida
$$
\theta_j:=\theta_j\left(1-\alpha\frac{\lambda}{m}\right)-\alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^i)-y^i)x^i_j
$$
Usualmente $(1-\alpha\frac{\lambda}{m})$ é menor que 1. Assim essa primeira parcela diminui $\theta_j$ (pra 0.99 do valor por exemplo), aproximando-o de 0 a cada iteração. Diminui-se então o parametro a cada iteração, e é feita sobre essa atualização a regulariação.
$$
\theta = (X^TX+\lambda A)^{-1}X^Ty\\
A = \begin{vmatrix}0&0&\dots&0\\
0&1&\dots&0\\
\vdots&\vdots&\ddots&0\\
0&0&\dots&1
\end{vmatrix}
$$
Com $X$ sendo $m\times (n+1)$, $y: m\times1$ e $A:(n+1)\times(n+1)$, tipo uma matriz identidade com primeiro elemento nulo. A regularização ainda faz com que sempre tenha uma inversa, para o termo entre parenteses, tendo outra vantagem para sua implementação.

### Na regressão logística

A função de custo é modificada para:
$$
J(\theta)=-\left(\frac1m\sum_{i=1}^m y^i\log(h_\theta(x^i))+(1-y^i)\log(1-h_\theta(x^i))\right)+\frac\lambda {2m}\sum_{j=1}^n\theta_j^2
$$
E o update de $\theta$ se torna "igual" ao da regressão linear:
$$
\begin{cases}
\theta_0&:= \theta_0-\alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^i)-y^i)\\
\theta_j&:= \theta_j-\alpha\Big(\frac{1}{m}\sum_{i=1}^m(h_\theta(x^i)-y^i)x^i_j+\frac\lambda m \theta_j\Big)
\end{cases}
$$


# Unsupervised Learning

Não se tem o output pros dados de entrada. Existe um grande banco de dados, mas não se sabe classificar ou separá-los. Tem-se uma mínima ideia, ou nenhuma, de como deveria ser o output dos dados. Não se sabe o efeito das variáveis.

## Clustering

Separa os dados em *clusters* (categorias). Exemplo: pegar um banco de dados de DNA e estimar que a concentração de algum gene relaciona com a idade, localização, etc.

## Non-Clustering

Exemplo: cocktail party - separar vozes a partir de gravações diferentes do mesmos sons