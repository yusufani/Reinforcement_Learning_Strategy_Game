# Reinforcement_Learning_Strategy_Game

YUSUF ANI


# 2020

Reinforcement Learning Tabanlı Strateji Oyunu

Yapay Zeka Projesi

# İçindekiler

[1-) Giriş 3](#_Toc40214563)

[2 - Modül -1 3](#_Toc40214564)

[2-1 Sonuç 4](#_Toc40214565)

[2-2 Modül-2 5](#_Toc40214566)

[2.2.1- Ortam Tasarımı 5](#_Toc40214567)

[2.2.2 – Sonuç 8](#_Toc40214568)

[2.3 – Modül-3 9](#_Toc40214569)

[2.3.1- Ortam Tasarımı 9](#_Toc40214570)

[2.3.2- Sonuç 11](#_Toc40214571)

[2.4 Modül -4 12](#_Toc40214572)

[2.4.1 – Ortam Tasarımı 13](#_Toc40214573)

[2.4.2 – Sonuçlar 14](#_Toc40214574)

[3- Genel Sonuçlar 14](#_Toc40214575)

[4- Kaynakça 15](#_Toc40214576)

# 1-) Giriş

Günümüzde hemen hemen her alanda yapay zeka modellerinin etkilerini görmekteyiz. Projemizde yapay zekanın alt alanlarından olan Reinforcement Learning&#39;i bir strateji oyununda uygulayarak hangi etkenlerin daha başarılı modeller oluşturduğunu ve ne gibi kısıtlarımızın olabileceğini araştırıldı. Projede Reinforcement Learning&#39;in uygulanması için popüler yaklaşımlar olan Q-learning ve Deep Q Learning kullanılmıştır.

Projede sonucunda oluşan kodlarda sadece şablon olarak linkteki[1] kodlar kullanılmıştır. Modelin ortamının tasarımı ve kodlanması ve networkünün tasarımı ve kodlanması bize aittir.

Projenin 4 farklı alt modülü vardır. Bu modüller arasına oyunun tasarımı ve networkün tasarımı değişmektedir. Her modülün altında modülde kullanılan oyunun tasarımı hakkında bilgi verilmiştir. Genel olarak bilgi vermek gerekirse oyunumuzda Agent olarak adlandırdığımız bir veya daha fazla yapay zeka modeli/modelleri karşısında belirli algoritmalarla yapay zeka olmadan oynayan bir Enemy oyuncusu bulunmaktadır. Ayrıca her ikisininde de birer Base kaleleri bulunmaktadır. Bu kaleler yok edilmediği sürece oyuncular tekrar canlanabilmektedir. Her modülde farklı boyutlarda kare bir map kullanılmıştır.

Sistemin başarısını ölçmek için ödül/ceza sistemi kullanılmıştır. Agent yaptığı belirli aksiyonlara karşı oyunun kuralları gereğince belirli ödül veya cezalar almıştır. Ödül ve ceza değerlerini optimize etmek de büyük bir problem olmuştur. Her modülde İlgili ödül ve ceza değerleri bulunmaktadır.

Not olarak eklenmelidir ki proje yaklaşık 3 hafta sürmüştür. Elimizde yaklaşık 200&#39;den fazla model ve çok fazla sayıda log dosyası vardır. İlgili modüllerde yapılan işlemlerde genel olarak ne amaçlandığı ve sonuçlarından bahsedilecektir.

Projenin çalıştırılması için requirements.txt dosyasındaki kütüphanelerin yüklü olması zorunludur.

## 2 - Modül -1

İlk modelimizde amacımız Q Learning kullanarak bir Enemy oyuncusuna karşı başarı elde etmektir. Q Learning için gerekli Q-Table oluşturulmuş Bir bakıma yapılacak proje için fizibilite aşaması olmuştur.

#### 2.1 – Ortam Tasarımı

Q-Table oluşturulması için Enemy ,Agent oyuncuları ve Enemy Base&#39;in x ve y koordinatları durumlar olarak alınmıştır. Bu durumda x ve y değerleri mapi büyüklüğüne bağlı olmaktadır. Oluşturulan Q-Table RAM&#39;de saklandığı için çok büyük map büyüklüğünün seçilemeyeceğini görülmüştür. Kendi cihazımızda 10 \*10 büyüklüğünde bir map için neredeyse yarım gün mapi oluşturulması gerektiği görülmüş ve denemelerde 4 , 5 , 6 değerleri denenmiştir. Ayrıca buradan şu yorum çıkartılabilir ki Q – Learning gerçek hayatta hatta karmaşık durumlarda bile gereksinimleri dolayısıyla etkili bir yöntem olarak gözükmemektedir.

Ortamda 1 adet Agent oyuncusu ve 1 adet Enemy oyuncusu kullanılmıştır. Enemy oyuncusu yölendirilmesi için Enemy oyuncusu ile Agent oyuncusu ve Base&#39;in koordinatları arasında fark alınmıştır. Çıkan sonuçtan küçük olana gidecek şekilde bir yönlendirme sağlanmıştır.

Agent oyuncusunun yönetilmesi için Q-Table değerleri ile eğitimin başlarında epsilon greedy yöntemi kullanılmıştır. Burada başlarda epsilonun azalma katsayısını 0.975 yerinde 0.99 değerine çekmek daha başarılı sonuçlar vermektedir. Bunun nedeni ise Mapin küçük olmasından dolayı oyuncunun çabuk oyunu kaybedip çok fazla ulaşamadığı Q değerinin olması olarak yorumlanabilir.

Yukarıda bahsettiğim gibi karşıdaki modelin hazır olup iyi oynaması sebebiyle Agent oyuncusu eğitimi için gerekli adım sayısını bulamadığı farkedilmiştir. Bu sorun proje boyunca en büyük sorun olmuştur. Enemy modeli çok güçlü olması eğitimi imkansız hale getirirken , çok güçsüz olması durumunda ise model çok çabuk öldürmeyi öğrendiği için proje amacına ulaşmış olarak kaul edilmedi. Bu modüldeki çözümümüz ise Enemy oyuncusunun adımlarını bir sabitle kısıtlamak oldu . Rakip oyuncu her X adımda bir hareket ettirildi. X değeri bizim denemelerimizde 5 olarak denendi.

Ortam değerleri ve Ödüller:

SIZE = **5** # Map Size
EPISODES= **1000000** # Number of episode

BASE\_HEALTH= **10**
PLAYER\_ATTACK = **2** # Player attack power

MOVE\_PENALTY = **1** # Agent player get penalty for each step
ENEMY\_BASE\_HIT\_PENALTY = **5** # If enemy player hit to agent base
ENEMY\_PLAYER\_HIT\_PENALTY = **2** # If player hit to agent
BASE\_HIT\_PENALTY = **5** # If AGENT hit to ENENMY base
PLAYER\_HIT\_PENALTY = **2** # If AGENT hit to enemy player
WIN\_REWARD = **10 # LOSE REWARD = -1 \* WIN REWARD**

epsilon = **1**
EPS\_DECAY = **0.99999** # Epsilon greedy
SHOW\_EVERY = **4000** # Show Game in every SHOW\_EVERY STEP

LEARNING\_RATE = **0.1**
DISCOUNT = **0.95**

AGENT\_N = **1** # Number of agent
ENEMY\_N = **1** #Number of enemy

### 2-1 Sonuç

![](RackMultipart20200513-4-kv8ptw_html_e555d98942174d63.png)

Yukarıdaki grafik x ekseninde episode numarasını ve y ekseninde ise ödül değerlerini göstermektedir. Grafiğe bakılarak kazanma ödülünün 10 ve kaybetme ödülünün -10 olduğu durumlarda bu kadar yüksek ödül alması bizi de şaşırtmıştır. Modeli deneyimleyince modelin daha fazla puan almak için rakibi öldürmek yerine oyuncusunu sürekli öldürerek 2&#39;şer puan kazanmayı tercih etmiştir. Bu da bizim düşünmediğimiz bir durumdu :D

## 2-2 Modül-2

Her ne kadar Q learning güzel sonuçlar vermiş olsa da işi biraz daha karmaşık hale getirmek istedik. Bunun için tasarım kısmında çok fazla değişiklik gerekti.

### 2.2.1- Ortam Tasarımı

Bunun için öncelikli amacımız Deep Q-Learning algoritmasını kodumuza eklemek oldu. Bunun için bize birkaç Python yapısı haricinde bir Neural Network yapısı gerekti. Bunun için kullanımı en kolay olan keras kütüphanesini tercih ettik. Kodun eğitilirken eğitim bilgilerini görebilmek amacıyla da Tensorboard kütüphanesini kullandık.

Neural Network yapısını CNN mimarisi üzerine kurduk. Kısaca giriş katmanında modelin alacağı resim boyutu kadar bir input alan ve aksiyon sayısı kadar sonuç üreten bir model oluşturuldu. Modelin iç yapısı :

![](RackMultipart20200513-4-kv8ptw_html_6b09fed21697616.gif)

Neural network&#39;e verilen map resmi için renklerden oluşan bir temsil gerekiyordu. Bunun için şu şekilde bir resim oluşturduk:

![](RackMultipart20200513-4-kv8ptw_html_e0318ecd48967bee.png)

Burada beyazlar Agent oyuncusunu ve Base&#39;ini , griler Enemy oyunusunu ve base&#39;ini simgelerken , yeşiller ise bir sonraki paragrafta bahsedeceğimiz ormanlık alanı temsil ediyor. Ayrıca ayarlardan kaç tane ağaç olacağı ayarlanabiliyor ve bu ağaçlar rastgele olarak oluşuyorlar. Bu da modelin yolu ezberlemesinin önüne geçiyor.

Oyunu biraz daha karmaşık ve Agent&#39;ımızın daha stratejik düşünmesi amacıyla ormanlık alan eklemeye karar verdik. Eğer Agent ormanlık alana gitmeye çalıştıysa belirlenene ceza puanını verildi.

Oyuncu sınıfı ve base sınıfı ile ilgili değişiklikler yapıldı. Oyuncular Base&#39;lerinin önünde rastgele bir konumda canlanıyorlar. Aynı şekilde base&#39;ler de rastgele şekilde konumlandırılıyorlar. Oyuncular için aksiyon sayısı 9&#39;a çıktı. Bu aksiyonlar yukarı , aşağı , sağ , sol , sabit kalmak ve çapraz yönler olarak sıralanmıştır. Ayrıca düşman oyuncu için de yol bulma algoritmasının gelişmesi gerektiği farkına varılmıştır çünkü Öklid olarak uzaklığa giderken düşmanın da önüne ağaçlar çıkabilir ve oyunun gerçekçiliği bozulur. Bu yol bulma algoritması için Breath First Search seçilmiştir. BFS algoritmasını seçmemizin sebebi en kolay implemente edilmesi çünkü zaman anlamında kısıtımız vardı. Belki bu algoritma ileride A\* ile değiştirilebilir. Algoritma çalışamadan önce yine en yakın birim bulunuyor ardından o birim için bir BFS algoritması çalıştırılıyor.Ardından bulunan path&#39;den geri gelinerek bir sonraki adım bulunuyor . Buna göre Enemy oyuncusu yönlendiriliyor.

Ortam değerleri ve Ödüller:

SIZE = **10** # MAP SIZE
OBSERVATION\_SPACE\_VALUES = _(_SIZE **,** SIZE **,**  **3** _)_ **,** # 4 ,
ACTION\_SPACE\_SIZE = **9** # Number of Action
RETURN\_IMAGES = True #
MOVE\_PENALTY = - **1** #
FOREST\_PENALTY = - **10** # If the agent goes to the tree
p = _{_
&quot;FOREST\_AREA&quot; : **8**  **,**

&quot;AGENT\_TO&quot;: _{_&quot;ENEMY\_PLAYER&quot;: **25**** , **&quot;ENEMY\_BASE&quot;:** 100 **_}_** ,** # rewards for If agent attack enemy
&quot;ENEMY\_TO&quot;: _{_&quot;AGENT\_PLAYER&quot;: - **25**** , **&quot;AGENT\_BASE&quot;: -** 100 **_}_** ,** # rewards for If enemy attack agent

&quot;LOSS&quot;: - **500**** ,**
&quot;WIN&quot;: **500**** ,**

&quot;BASE\_LEN&quot;: **1**** ,**

&quot;AGENT\_PLAYER\_HEALTH&quot;: **100**** ,**
&quot;AGENT\_BASE\_HEALTH&quot;: **1000**** ,**
&quot;AGENT\_PLAYER\_ATTACK\_POINT&quot;: **150**** ,**

&quot;ENEMY\_PLAYER\_HEALTH&quot;: **100**** ,**
&quot;ENEMY\_BASE\_HEALTH&quot;: **1000**** ,**
&quot;ENEMY\_PLAYER\_ATTACK\_POINT&quot;: **50**
_}_

# HYPERPAREMETERS
 # Environment settings
EPISODES = **20\_000**
MIN\_REWARD = - **200** # For model save
 # Exploration settings

epsilon = **1** if model\_path==&quot;&quot; else **0.1**
#epsilon = 1
 # not a constant, going to be decayed
EPSILON\_DECAY = **0.99975**
MIN\_EPSILON = **0.001**

# Stats settings
AGGREGATE\_STATS\_EVERY = **1** # episodes
SHOW\_PREVIEW = True

env = Env_(_SHOW\_PREVIEW_)_

# For stats
ep\_rewards = _[_- **200** _]_

# For more repetitive results
random.seed_(_ **1** _)_
np.random.seed_(_ **1** _)_
tf.set\_random\_seed_(_ **1** _)_

###

Burada map büyüdüğü için epsilonu eski değerine geri getirdik. Ayrıca bundan sonraki modellerin aynı başlangıç noktalarından başlatarak daha karşılaştırılabilir sonuçlar elde ettik.

### 2.2.2 – Sonuç

![](RackMultipart20200513-4-kv8ptw_html_e2ee094250d87276.png)

Yukarıdaki grafikte alınan ödül miktarında bir artış gözüküyor. Bu grafik için bir smooting uygulanmıştır. Burada maalesef eğitimler için Google Colab ortamına geçtiğimiz için 11k ya kadar ki kısım kaydedilmiş. Biz modeli 20 k adım eğittik ve modelin başarısı arttı .

Modeli test ettiğimizde modelin karşı rakibin geldiği yeri tahmin ederek onu öldürerek puan kazandığını gördük. Aslında model amacımızın dışına çıktı fakat yine de oyun kuralları içerisinde kaldığı için başarılı olarak gördük.

## 2.3 – Modül-3

Bu modülde işi bir adım öteye taşımaya karar verdik. Yapmak istediğimiz şey bir ajanın birden fazla oyuncuyu aynı anda kontrol etmesi. Bu durum ise en çok zamanımızı harcayan durum oldu.

### 2.3.1- Ortam Tasarımı

Bu modelde önceki modelin yakaladığı oyun ile ilgili hataları düzeltmekten ve networkümüzü değiştirmek haricinde büyük değişiklikler yapmadık. Oyunu görselleştirmek için pygame kütüphanesi ve internetten bedava iconlar bulduk . Eski yaklaşımla beraber Network konusunda da yeni bir yaklaşım kullandık: İnternette araştırdığımızda çoğu Reinforcement Learning CNN mimarileri üzerinde kurulmuştu. Bunun nedeni ise genelde Reinforcement Learning ile yapılmış oyunlarda insanla bir karşılaştırma içine giriliyor. İnsanlar durumları gözetmek için sadece göz ile veri alabildiği için modeli için de aynı durum düşünülmüş. Bizim oyunumuzda insanın oynaması için şimdilik bir arayüz yok. Bu yüzden biz Fully Connected Layerlarla da sonuç alabileceğimizi düşündük.

Yeni networkümüzün yapısı şu şekilde oldu :

Input Size : Map + Oyuncuların Birbirilerine uzaklıkları + Oyuncuların ve baselerin canları

![](RackMultipart20200513-4-kv8ptw_html_37e452b7d81edcc.gif)

Output olarak ise (agentın yönettiği oyuncu sayısı \* Her bir oyuncunun aksiyon sayısı ) aldık . 3 agentlı 9 aksiyonlu bir oyun için 27 observation değeri olmuş olur .Q değerlerinin güncellemek için ise ilk agent için ilk 9 değeri alıp maksimun olanı seçilip aksiyon olarak verildi.Yeni durum modele tekrar verilip gelen değerler arasında ilk 9&#39;u alınarak Deep Q learning denklemi ile değer hesaplandı. İkinci agent ve üçüncü agent için ise kendilerine denk gelen sonuçlar alınıp değerler güncellendi.

Bunlara ek olarak CNN modeli ve Fully Connected Layerlar ayrı ayrı denenerek performans karşılaştırması yapıldı.

![](RackMultipart20200513-4-kv8ptw_html_3fb6d46bc3e0e22f.png)

Ortam değerleri ve Ödüller:

NUMBER\_OF\_AGENT\_PLAYER = **3**
NUMBER\_OF\_ENEMY\_PLAYER = **3**
SIZE = **10**
#OBSERVATION\_SPACE\_VALUES = SIZE\*SIZE+(NUMBER\_OF\_AGENT\_PLAYER+1)\*(NUMBER\_OF\_ENEMY\_PLAYER+1)+NUMBER\_OF\_AGENT\_PLAYER+NUMBER\_OF\_ENEMY\_PLAYER+2 # +2 for base health
OBSERVATION\_SPACE\_VALUES = _(_SIZE **,** SIZE **,**** 3**_)_

ACTION\_SPACE\_SIZE = **9**

MOVE\_PENALTY\_AGENT\_AREA = - **2**
MOVE\_PENALTY\_ENEMY\_AREA = **1**

FOREST\_PENALTY = - **20**
p = _{_
&quot;FOREST\_AREA&quot; : **8**  **,**

&quot;AGENT\_TO&quot;: _{_&quot;ENEMY\_PLAYER&quot;: **40**** , **&quot;ENEMY\_BASE&quot;:** 80 **_}_** ,**
&quot;ENEMY\_TO&quot;: _{_&quot;AGENT\_PLAYER&quot;: - **20**** , **&quot;AGENT\_BASE&quot;: -** 50 **_}_** ,**

&quot;LOSS&quot;: - **500**** ,**
&quot;WIN&quot;: **1000**** ,**

&quot;PLAYER\_N&quot;: **1**** ,** # player key in dict
&quot;FOOD\_N&quot;: **2**** ,** # food key in dict
&quot;ENEMY\_N&quot;: **3**** ,** # enemy key in dict
&quot;BASE\_LEN&quot;: **1**** ,**

&quot;AGENT\_PLAYER\_HEALTH&quot;: **100**** ,**
&quot;AGENT\_BASE\_HEALTH&quot;: **1000**** ,**
&quot;AGENT\_PLAYER\_ATTACK\_POINT&quot;: **150**** ,**

&quot;ENEMY\_PLAYER\_HEALTH&quot;: **100**** ,**
&quot;ENEMY\_BASE\_HEALTH&quot;: **1000**** ,**
&quot;ENEMY\_PLAYER\_ATTACK\_POINT&quot;: **50**
_}_

### 2.3.2- Sonuç

![](RackMultipart20200513-4-kv8ptw_html_2ed21913252262db.png)

![](RackMultipart20200513-4-kv8ptw_html_fca992df5cdb9b62.png) ![](RackMultipart20200513-4-kv8ptw_html_66fd8e1d8880559b.png)

Yukarıdaki grafiklerde 5şer adımın sırıyla ortalama ödül , maksimun ödül ve minimun ödül değerleri gözükmektedir. Mavi çizgi CNN modelini temsil ederken turuncu renk temsil etmektedir.

Öncelikle modeller arasında çok fark olmadığı gözükmüştür fakat biraz da olsun Fully connected layerlı modelin daha önde olduğu gözüküyor. Ayrıca grafikler incelendiğinde modellerin başarı oranlarında büyük değişiklikler olmadığı gözüküyor. Bu yüzden Modül -3 denemeler sonucunda başarısız olmuştur.

Başarısızlığın nedenini araştırınca multi agent Reinforcement Learningin daha farklı uygulandığını başlı başına bir konu olduğunu ve üstünde çalışmaların devam ettiğini gözlemledik.

## 2.4 Modül -4

Buraya kadar olan modellerle ödev için kriterleri sağladığımızı düşünüyoruz fakat aklımıza diğer modellerden çok daha eğlenceli ve farklı bir yapı denemeye karar verdik.

### 2.4.1 – Ortam Tasarımı

Burada aslında 2. Modülün çok benzerini yaptık. Farklı olarak elimizde 1 tane Agent modeli yerine 3 tane Agent modeli var ve ortak olarak Enemy modeli bulunuyor. Ayrıca Neural Network yapısında da biraz daha performanslı olduğuna inandığımız Fully Connected Layerlarla oluşturduk. Ayrıca modellere daha fazla alan vermek için 10 olan map boyutunu 20&#39;ye çıkardım. Ayrıca son kullanıma hazırlamak üzere arayüzü ve bilgilendirme ekranlarını düzelttik .

Neural Network yapısı :

Input : ( Map Size \* Map Size + İlgili Agentın diğer tüm Agent ve Enemy birimlerine uzaklıkları + Tüm Agent ve Enemy birimlerinin canları )

Output : 8 Adet aksiyon ( aksiyonlar yukarı , aşağı , sağ , sol ve çapraz yönler olarak sıralanmıştır)

![](RackMultipart20200513-4-kv8ptw_html_397afef6de128923.gif)

Ortam değerleri ve Ödüller:

Önceki modellerden farklı olarak daha fazla saldırıya yatkın modeller yapmak için ödül değerlerinde oynamalar gerçekleştirdik. Bu oynamaları aşağıdaki belirtilmiştir:

Modellere yeni olarak

Map dışına çıkması denemesi halinde -10 puan ceza verildi.

Modele en yakın Base&#39;in Agent oyuncusuna ulan uzaklığı ile orantılı bir ödül sistemi kuruldu:

Ödül = Map boyutu / 2\*uzaklık

Eğer Agent oyuncusu base&#39;e yakın ise yüksek başarı elde edecektir.

Artık ağaçların koordinatları kullanıcıdan alınabiliyor.

##

## 2.4.2 – Sonuçlar

Modelleri başarını ölçmek için reward değerlerini göstermek pek uygun olmayacağını düşündük. Çünkü ortam sabit olmayacağı için modellerin başarısı karşıdaki modellerin başarısı ile ters orantılı olduğunu gördük.

Sonuçlardan kesin olarak çıkarabileceğimiz konuş ise modellerin hepsinin başarısının zamanla artması oldu.

Bir başka gözlemimiz ise modelleri sadece 2500 bölüm eğitmemiz oldu. Önceki modellerede bu değer 20000&#39;di. Bunun da en büyük sebebi oyunun bitmesine kadar modellerin daha fazla zamanı olması olarak görülebilir.

Modellerdeki arasında if komutlarıyla yönetilen modelle en çok karşılaşanın en çok başarıyı aldığını gördük.

## 3- Genel Sonuçlar

1. Biz çalışmalarımızda Deep Q learning kullandık. Hazır bir aksiyon , araba yarışı veya basit labirent tarzı oyundaki performansının çok iyi olduğu biliniyor. Hazır bir takım yapay sinir ağları ile uygulanabilirliğini gördük. Fakat şu bir gerçek ki strateji gibi karmaşık oyunlarda Deep Q Learning&#39;in üstüne başka algoritmalar da eklenmesi buna göre yapay sinir ağı tasarlanması gerekebilir. Yine de küçük ölçek için öğrenebilme yetisinin olduğunu gördük.
2. Ödüllerin ne olduğu ve ne kadar olduğu çok fazla farkediyor. Çoğu durumda modelin bir ödülü almanın kısa yolunu bulup ezberlediğini görüldü.
3. Reinforcement Learning Ajanlarının oyunlardaki hataları bulup faydalanması diğer oyuncuların bulmasına oranla çok daha hızlı oluyor. Bug bulma alanında kullanılabilir.
4. Projedeki gibi karşıda bir rakibin olduğu durumlarda rakibin hamleleri çok fazla etki ediyor. Rakip ne çok güçlü olmalı ne de çok güçsüz olmalı.

## 4- Kaynakça

1. https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/
