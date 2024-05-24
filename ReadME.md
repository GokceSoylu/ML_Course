# Uni Notes

Merhabalar,  

Bu bir intihar mekubudur 😊 

 Şaka şaka intoduction to machine learning dersine çalışırken aldığım notlar. Dersin özeti, hap bilgi olarak düşünebilirsiniz. 

![Alt text](<Screenshot 2024-05-22 at 07.45.58.png>)

Hocam makine öğrenmesi dediğimiz zaman. Makielere öğretmekten bahsediyoroz. Mutfağınızda hangi makine varsa blender olur mikro dalga olur alın karşınıza başlayan hayatı öğretmeye 😬 yaptığım son iğrenç espriydi. Ciddiyetle anlatıyorum hemen 

 

Makine öğrenmesini üç kısıma aayırıcaz supervised(gözetimli) leraning, unsupervised(gözetimsiz) learning ve Reinforcement learning. Bu ders kapsamında daha çok ilk ikisi üzerinde durucaz.  

### SUPERVİSED LEARNİNG 

Etiketli verilerin kullanıldığı modellerdir. Modeli eğitmek için kullandığımız veri etiketli olur. Zaten bu yüzden supervised yani gözetimli öğrenme diyoruz. Verdiğimiz verinin sınıflandırması (etiketlenmesi) insan gözetimi altında istediğimz gib, sınıflayıp veriyoruz veriyi. Bu yüzdende supervised learning için “ such as an input  where desired output is knonw” tam olarka mevzu gözetimli öğrenmede zaten veriyi sınıflandırıp laballayıp. Sonrada modeli hazır bizim sınıflandırdığımız veryle eğittiğimizde isteğimz çıktıyı. Yani yenş veriyi nasıl nereye koymasını istediğimizi zaten çok iyi biliyoruz.  

### Unsupervised Learning 

Geldik zor kısma. Hocam yukarıdakinden mevzuya biraz uyandık zaten supervised de labellı datayı training date set olark kullanıyorsak unsupervised learning de labelsız veriyi once sınıflandırıp sonr atraining set olarak kullanıcaz. Yani labellama-sınıflandırma kısmıda makineye kalıcak. Gözetimsiz öğrenmeyi karmaşık yapan kısımda aslında eğitim setnde etiketi olmayan veriyi nasıl sınıflandıracağımız. iIleride bunun üzerinde bol bol durucaz. 


![Alt text](<Screenshot 2024-05-22 at 07.46.24.png>)

## TERMİNOLOJİ 

Hocam şimdi lineaer regresyon üzerinden anlatıcam kavramları çünkü daha kolay oluyor. 

Y=ax1+a2x2+a3x3+- - - - +anxn 

Label : label dediğimiz şey etiket. Biz modeli eğittikten sonra yeni bir veri verip bu verinin yerini yani etiketin bulmaya çalışmıyor muyuz? Yani label dediğimiz şey denklemde y ye takabül ediyor. 

Feature : bunlar verinin özellikleri. Sınıflandırmak için kullanılan özellikler yani. Nasıl y yi bulmak için x değikenşne veriden değer etkiliyorsa yanş x lerle işlem yapıp y yi bulıyorsak veride de feture yani özelliklerine bakıp label’ını buluyoruz. Mesela bir yaprağın türünü bulmak istresdiğimizi düşünelim. Yaprağın genilğişi, rengi, kalınlıı feature olur yani X. yaprağın türü de y yani label olur 😊 hadi ek bilgi y değerini bulmak için daah önemli olan feature’lara yani x değerlerine dah yüksek katsayı veririz. Yani modeli eğitirken önemli olan özelliklerin ağırlığını narttırıcaz yani dah yüksek a değeri vericez 😊 

### Regresyon ve Classification 

Regresyonda sürekli değerlerle ilgilenir. Bu reklema tıklanma ihtimali. Karabükte 10 yıl sonra  bulunacak ev sayısının tahmini vb. 

Sınıflandırma ise süreksiz değerler bakar e-mail spam yada değil. Bu cisim köpek, ekmek yada domuz gibi. Burada süreksizden kastımız 0-1 olması sürekliden kastımız normal sayı olması. Yani grafiğe baktığın sınıflandırma 0-1 şeklinde olacağı için kesikli bir grafik elde edilir. Diğerinde ise sayı olcağı için sürekli grafik olur. Sürekli süreksiz muhabbeti buradan gelir. 

-> Başlıca Sıkıntılar 

Böyle başlık mı olur? Hocam ilerde üzerinde bol bo duracağımız ml’de sık karşılalaşılan sorunlara bir üsten göz atalım. Rakibe bi selam verelim gibi 

* Insufficient Quantity of trainin data: yani veri miktarı yetersizliği 
* Nonrepresentetive data: temsili olmayan veri. Elimizdeki verinin geneli temsil etmesi gerekir. Eğer verimiz çok az olursa gürültü olma ihtimali gürültüden etkilenme ihtimali daha yüksktir. Ancak data nın çok olması olması bizi temsil sorunundan direkt kurtarmak eğer örnekleme metodund ahata varsa yine verimiz yanlı olabilir(samling bias) 
* Poor qualty data: hocam eğer verimizde yanlış ölçüm nedeiyle bol bol error(hata), outliers(aykırı değer), noise(gürültü) varsa bu durumda verimizi temizlememiz gerekir. Bu durumda ya bu özelliği görmezden geliriz ya boş kısımları median(ortanca) ile doldururuz. 
* Irrelevant features: ilgisiz özellikler. Hocam burada feature selection(özellik seçimi) ve feature extraction(özellik çıkarma) durumları ortaya çıkıyor. Iyi veris seti koy abicim koy bu özelliği de ne bulursan ekle diyerek çıkmaz. Elimizdeki özelliklerden iyi,  gerekli olanları seçmek ve gereksiz olanları çıkarmalıyız. 
* Overfitting: makinemiz ğitim setine mükemmel uyum sağlar ve çok yüksek oranda doğruluk oranınan sahiptir. Ancak başk averi verildiğinde başarı ciddi şekilde düşer.  
* Underfitting :overfittingin tersi oluyor. Modelimiz basit kaçarsa olur. Yine doğruluk oranı haliyle düşük olur. Modelimiz başarılı olamaz. Bunun için data ile değil modelimiz ile uğraşmamız gerekir. Daha çok parametreli bir model seçebiliriz yada model üzerindeki kısıtlamaları azaltabiliriz vb. 
 
### Testing And Validation 
Hocam veri setinin tamamıyla modeli eğitmessin önce bir kısmını valide (doğrulama) etmek için yani modelini test için ayırırsın. Yanş verinin bir kısmıyla model eğitilir kalan kısmıyla valide edilir.  
 
### Hyperparameter Tuning 
--bu kısım daha sonra eklenecektir-- 
 