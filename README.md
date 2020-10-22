# Smoke_detection
  Раннее обнаружение источника возгорания, точная локализация и принятие своевременных мер по тушению является актуальной задачей. Своевременное обнаружение и принятие соответ-ствующих мер имеет решающее значение для предотвращения катастроф, что влечет за собой спасения жизней и сохранение имущества людей. 
  Мы повседневно сталкиваемся с системой обнаружения пожаров в виде датчиков огня и дыма. Они широко используются в помещениях и обычно требуют, чтобы огонь горел в течение некото-рого времени, чтобы образовалось большое количество дыма, а затем сработала сигнализация. Кроме того, эти устройства не могут быть развернуты на открытом воздухе в больших масштабах, например, в лесу.
Существует большое количество решений по детектированию пожароопасных объектов с использованием беспилотных летательных аппаратов, в том числе и с использованием алгоритмов машинного обучения [1-2]. В данной работе рассматривается система, обрабатывающая данные с видеокамер, расположенных в лесной зоне. 
  Система обнаружения пожара на основе технического зрения захватывает изображения с камер и немедленно обнаруживает возгорание, что делает их пригодными для раннего обнаружения пожара. Такая система также дешева и проста в установке. В этой работе мы предложили метод обнаружения пожара на основе компьютерного зрения, который может работать со стационарной камерой.
  Система видеонаблюдения на базе камеры может контролировать указанную территорию в реальном времени с помощью обработки видео. Когда система производит обнаружение пожароопасного объекта, она отправляет захваченное изображение тревоги администратору. Администратор делает окончательное подтверждение на основе отправленного изображения тревоги.

# Общее описание решения
Система обнаружения пожара захватывает видео с камеры и разбивает его на фреймы для дальнейшей обработки. Для сохранения признаков динамики используется алгоритм вычетания фона. Обработанные кадры поступает на сверточную нейронную сеть EfficientDet-D1, обученную находить участки возгорания. В качестве дополнительной проверки, обнаруженный ранее участок отправляется на классифкатор, представляющий собой  сверточную нейронную сеть на основе сетей краткосрочной памяти (англ. Long short-term memory, LSTM). Результатом является ограничивающая рамка подтвержденная классификатором. Конечным этапом обработки является алгоритм кластеризации, отмечающий эпицентр возгорания.

# Общее описание логики работы решения
Тут можно привести блок-схему
 Ну а тут можно увидеть тестовую картиночку
 
  ![Scheme](https://github.com/Vladislav26Laptev/Smoke_detection/blob/main/%D0%A1%D1%85%D0%B5%D0%BC%D0%B0.png)

# Данные
Текст

# Алгоритм обработки видеоряда 
Из-за динамического характера пожара, форма дыма и пламени неправильная и постоянно меняется. Поэтому при использовании дыма в качестве важного признака для обнаружения движения, обычными методами обнаружения, являются: непрерывная смена кадров [3], вычитание фона [4] и моделирование смешанного фона по Гауссу [5]. Вычитание фона необходимо для правильной установки фона, потому что между днем и ночью большой промежуток. В общем, иметь константу сложно, и для нее нужно задавать параметры, что сложнее статического фона. Смешанная гауссовская модель слишком сложна и требует установки исторического кадра, числа гауссовской смеси, частоты обновления фона и шума на этапе предварительной обработки, поэтому этот алгоритм не подходит для предварительной обработки, так как мы ориентируемся на съемку одного направления в 14 секунд. Преимущество метода разности кадров - простота реализации, низкая сложность программирования, нечувствительность к изменениям сцены, например, к освещению, и возможность адаптации к различным динамическим средам с хорошей стабильностью. Недостатком является невозможность извлечения всей площади объекта. Внутри объекта есть «пустая дыра», и можно выделить только границу. Поэтому в данной статье принят улучшенный метод разности кадров. Поскольку поток воздуха и свойства самого горения будут вызывать постоянное изменение пикселей пламени [6], пиксельные изображения без огня могут быть удалены путем сравнения двух последовательных изображений. Мы используем улучшенный алгоритм разности кадров. Сначала видеопоток преобразуется в последовательность кадров. Далее, над кадрами с определенным интервалом выполняется преобразование из трех каналов RGB в один канал (переход в градации серого), что экономит время вычислений. На следующим шаге выполняется операция инициализации «усредненного кадра» (1). Для других изображений используется отличие кадра от «усредненного». Формула разности кадров приведена в (2). 

F_с (x,y)=1/N*∑_(i=1)^N▒〖F_i (x,y)〗	(1)

F_(d_i ) (x,y)=|F_c (x,y)-F_i (x,y)|,i=1,2,..N,	(2)


где F_с (x,y) – «усредненный кадр», N – общее число, обрабатываемых, кадров. F_i (x,y) – текущий кадр последовательности. F_(d_i ) (x,y) – разность текущего кадра последовательности и усредненного. F_(r_i ) (x,y) – результирующий кадр операции шумоподавления.
В текущей работе рассматривается запись со статичной камеры в 10 секунд. Из видеоряда, разбитого на кадры, извлекаются 7 кадров с равным интервалом по времени. На выходе ожидается 4 обработанных кадра с номерами 1, 3, 5, 7 для более точного обнаружения. Результат обработки представлен ниже (см. рис. 1).

 	 	 
(a)	(b)	(c)

# Алгоритм обнаружения объекта
После алгоритма предварительной обработки, каждый полученный кадр последовательно обрабатывается моделью распознавания объектов EfficientDet-D1. Общая архитектура EfficientDet [11] в значительной степени соответствует парадигме одноступенчатых (one-stage) детекторов. За основу взята модель EfficientNet, предворительно обученная на датасете ImageNet. Отличительной особенностью от одноступенчатых детекторов [12, 13, 14, 15], является дополнительный слой со взвешенной двунаправленной пирамидой признаков (BiFPN), за которым идёт классовая и блочная сеть для генерации предсказаний класса объекта и ограничивающего прямоугольника (бокс) соответственно. Бокс имеет четыре параметра, координаты (x,y) для верхнего левого угла и координаты для нижнего правого угла. Для обучения сети требуются кадры с нанесенной разметкой в виде боксов с указанием соответствующего класса. 
Так же были рассмотрены модели обнаружения объектов: SSD MobileNet v2, Faster R-CNN ResNet50 V1, Faster R-CNN Inception ResNet V2. Все модели обучались в одинаковых условиях, для сравнения использовались 2-а критерия: точность (accuracy) (4), скорость работы (speed, время обработки одного кадра), результат приведен ниже (см. табл. 1).

accuracy=(TP+TN)/(TP+TN+FP+FN)	(4)

где TP (True Positive) – в кадре обнаружен настоящий дым, FP (False Positive) – дыма нет, но есть обнаружение, TN (True Negative) – дыма нет, и не нет обнаружения, FN (False Negative) – настоящий дым, нет обнаружения.

Таблица 1
Сравнение результатов моделей object detection 
Model name	Accuracy, %	Speed, s
EfficientDet-D1	96	0,64
SSD MobileNet v2	83	0,59
Faster R-CNN ResNet50 V1	89	1,6
Faster R-CNN Inception ResNet V2
91	2,1


# Алгоритм классификации
Текст

# Алгоритм постобработки
Текст

# Итог
Текст

# Источники
1. C. Yuan, Z. Liu and Y. Zhang,” UAV-based forest fire detection and tracking using image processing tech-niques”, Proceedings of 2015 International Conference on Unmanned Aircraft Systems (ICUAS), 2015, pp. 639–643.
2. G. N. Rao, P. J. Rao and R. Duvvuru,” A drone re-mote sensing for virtual reality simulation system for forest fires: semantic neural network approach”, Proceedings of IOP Conference Series: Materials Science and Engineering, vol. 149, no. 1, 2016, p. 12011.
3. T. Song and L. Pan, “Spiking neural P systems with learning functions,” IEEE Trans Nanobioscience, vol. 18, no. 2, pp. 176–190, 2019. 
4. A. Aggarwal, S. Biswas, S. Singh, S. Sural, and A. K. Majumdar, “Object tracking using background subtrac-tion and motion estimation in MPEG videos,” in Proceed-ings of the Asian Conference on Computer Vision, Springer, Berlin, Germany, January 2006. 
5. T. Song, X. Zeng, P. Zheng, M. Jiang, and A. Rodr´Iguezpaton, “A parallel workflow pattern modelling using spiking neural P systems with colored spikes,” IEEE Transactions on Nanobioscience, vol. 17, no. 4, pp. 474–484.
6. W. Yang, M. Mortberg, and W. Blasiak, “Influences of flame configurations on flame properties and no emis-sions in combustion with high-temperature air,” Scandinavi-an Journal of Metallurgy, vol. 34, no. 1, pp. 7–15, 2005.
