

# Table of Contents

1.  [task, Задача](#orgfa55554)
2.  [analysis, Анализ задачи](#org7975d86)
3.  [learned tems, Изученные термины](#org68dde9a)
4.  [1) High-frequency trading in a limit order book](#org17c4c14)
5.  [4) Short-Term Market Changes and Market Making with Inventory](#org108c6cf)
6.  [answer:](#org5343c4c)

NFT Data Engineer, HFT(High-frequency trading)


<a id="orgfa55554"></a>

# task, Задача

1.  Avellaneda Stoikov, 2008 <https://www.math.nyu.edu/~avellane/HighFrequencyTrading.pdf>

Классическая статья по mm. Большинство из современных статей по mm так или базируются на ней или хотя бы ссылаются.

1.  Optimal Strategies of High Frequency Traders, 2015
    <https://scholar.princeton.edu/sites/default/files/JiangminXu_JobMarketPaper_Revised_0.pdf>

На базе Avellaneda Stoikov было написано много статей, вот одна из них. Она добавляет ряд интересных
 вещей по направленности рынка, имбалансам ордербуков, ping-ордерам и т. д. Все эти вещи приближают
 исследования к реалиям рынка.

Однако перевести ту логику в код задача не из простых. К счастью, есть хорошая серия постов с
 разбором статьи: 3) <https://www.quantalgos.ru/?p=51> За все время это чуть ли единственный случай,
 когда на русском мы нашли что-то более серьезно проработанное, чем на английском. Они там во много
 заходов правят код, в комментариях постятся какие-то баги, но в целом что-то более-менее рабочее у
 них получается.

1.  Short-Term Market Changes and Market Making with Inventory, Jin Gi Kim, Sam Beatson, Bong-Gyu
    Jang, Ho-Seok Lee, and Seyoung Park, 2020
    <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3639715> (SSRN-id3639715.pdf)

Еще одна статья на базе Avellaneda Stoikov, которая развивает их идею в другом направлении. Нужно
 будет понять суть этого направления и совместимо ли оно с идеями из статьи выше.


<a id="org7975d86"></a>

# analysis, Анализ задачи

Avellaneda Stoikov - два автора статьи 1)

Avellaneda Stoikov - is two authors of article 1)

1,2,3 - одно направление, 4 - другое.

1,2,3 - one subject, 4 - another one.

Задача понять суть второго направления и совместимо ли оно с идеями какой-то статьи выше 1, 2 или 3.
The task is to understand second subject and judje how ideas from second compatible with article 1,2 or 3.

Задачи и их подзадачи. Tasks and their subtasks.

-   To understand second subject from article 4.
    -   to understand first subject from 1, 2 and 3.
-   To judge compatibility of second subject with 1,2 or 3 article
    -   to understand second subject
    -   to choose best article for unclear formulation.
    -   to judge difference and compatibility

-   Понять вторую тему из статьи 4.
    -   понять первый предмет из 1, 2 и 3.
-   Оцените совместимость второго предмета с 1,2 или 3 статьей
    -   понять второй предмет
    -   выбрать лучшую статью за неясную формулировку.
    -   оценить разницу и совместимость


<a id="org68dde9a"></a>

# learned tems, Изученные термины

-   **order book** or **depth-of-market (DOM)** - стакан заявок
-   **limit order** is an order to buy a security at no more than a specific price, or to sell a security at
    no less than a specific price (called "or better" for either direction)
-   **inventory** - the stock of securities that a trader holds at any given time. By maintaining a
    certain level of inventory, traders can take advantage of market opportunities and quickly respond
    to changes in demand. However, holding too much inventory can also increase the risk of losses if
    the market moves against the trader.
-   A **call** option in stock trading is a contract that gives the holder the right, but not the
    obligation, to buy a specific underlying asset (such as a stock) at a predetermined price (the
    strike price) within a specified period of time.
-   **spread** - the difference between the bid price (the highest price a buyer is willing to pay for a
    stock) and the ask price (the lowest price a seller is willing to accept for the same stock). The
    spread can be used as an indicator of market liquidity and volatility.  narrower spread generally
    indicates a more liquid and stable market
-   **leg** - one part of a complex trade that involves multiple transactions. For example, if a trader
    wants to buy a stock and sell a call option on that same stock, those two transactions would be
    considered two legs of the overall trade.
-   **aggressor** in HF trading - trader who make order some distance away from the equilibrium price.
-   **split-second trading decisions** - просто скоростная торговля, ничего конкретного под этим термином нет


<a id="org17c4c14"></a>

# 1) High-frequency trading in a limit order book

sources of risk

-   the **inventory risk** from uncertainty in the asset's value (invertory effect)
-   asymmetric information risk arising from informed dealers

To account the effect of the inventory we specify "true" price for asset and derive optimal bid and
 ask quotes around this price.

Arrival rate of buy and sell orders is crucial. Model arrival rates.

Zero intelligence agents.

Optimal bid and aks quotes are derived in an intuitive two-step procedure.

1.  compute a personal indifference valuation for the stock, given his current inventory.
2.  calibrates his bid and ask quotes to the limit order book, by considering the probability with
    which his quotes will be executed as a fuction of their distance from the mid-price.

Our solution is the balancing act between the dealer's personal risk and the market environment

1.  оценка собственного безразличия к запасам
2.  вероятность реализации квот, как функция от их расстояния до mid-price(true price)

building blocks:

-   f(the distance to the true price) = target
-   to solve optimal bid and ask quotes, and relate them to the reservation price of the agent.
-   approximate solution
-   Profit and Loss (P&L) profile.


<a id="org108c6cf"></a>

# 4) Short-Term Market Changes and Market Making with Inventory

Closed form optimal bidding and asking strategies of the market maker.

if spread is wide traders submit limit orders because an aggressive buy or sell immediately is
 expensive.

stochastic investment opportunities - factor.

dynamic programming - utility

Poisson jump processes for modeling trading intensity and stock volatility as switching
 mechanism. (it is a key difference from 1) aricle.)

taking a utility maximizing approach ( same with article 1))


<a id="org5343c4c"></a>

# answer:

Я понял вопрос как:

Нужно понять суть направления в статье

Short-Term Market Changes and Market Making with Inventory, Jin Gi Kim, Sam Beatson, Bong-Gyu
 Jang, Ho-Seok Lee, and Seyoung Park, 2020
 <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3639715> (SSRN-id3639715.pdf)

и совместимо ли оно с идеями из статьи

Avellaneda Stoikov, 2008 <https://www.math.nyu.edu/~avellane/HighFrequencyTrading.pdf>
.

Суть статьи в расчете торгующим агентом цен bid и ask заявок, с учетом вероятности, что заявка будет
 выполнена, как функция растояния от истенной стоимости, при этом учитывается стоимость инветаря
 трейда.
Да, с идеями совместима, реализация другая. При написании статьи автор взял из статьи Avellaneda
 Stoikov подход, ориентированный на максимизацию полезности (taking a utility maximizing approach).
 Отличием являются Пуассоновские скачкообразные процессы для моделирования интенсивности торговли и
 волатильности. ( Poisson jump processes for modeling trading intensity and stock volatility as
 switching mechanism.), в то время как, в статье Avellaneda Stoikov не учитывается изменение
 инвестиционной перспективности.
