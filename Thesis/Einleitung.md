\chapter{Einleitung}

\section{Hintergrund der Arbeit}

Die Automatisierung von Prozessen in unterschiedlichen Branchen und die ständige Verfügbarkeit von Daten haben dazu geführt, dass die Verarbeitung und Analyse von großen Mengen an Texten immer wichtiger wird. Eine wichtige Aufgabe in diesem Zusammenhang ist die Erkennung von Eigennamen. Beispiele für Eigennamen sind Personen, Orte und Organisationen. Das Erkennen von solchen Namen nennt man Named-Entity-Recognition (NER) oder Eigennamen Erkennung im deutschen. Diese Informationen sind für die Textanalyse von großer Bedeutung.

Eine Herausforderung bei NER ist jedoch die Verfügbarkeit von ausreichend annotierten Daten. In vielen Anwendungen sind jedoch nur  ``weakly labelled'' , in deutsch ``schwach annotierte''   Daten verfügbar, bei denen lediglich die Anwesenheit von Eigennamen angegeben ist, jedoch nicht deren konkrete Kategorie oder andere Fälle von nicht korrekten Datenangaben. 

Die Bachelorarbeit mit dem Titel ``Named Entity Recognition with Weakly Labelled Data''  hat das Ziel, Methoden zu entwickeln und zu untersuchen, die es ermöglichen, NER-Aufgaben auch mit schwach annotierten Daten zu lösen. Durch die Untersuchung von verschiedenen Ansätzen und Algorithmen kann gezeigt werden, wie die Genauigkeit der Erkennung von Eigennamen durch die Verwendung von schwach annotierten Daten verbessert werden kann.

Diese Arbeit bietet daher nicht nur einen wichtigen Beitrag zur Verbesserung von NER, sondern auch die Möglichkeit, die Anwendbarkeit von Methoden zur Textanalyse unter realistischen Bedingungen zu untersuchen. In dieser Arbeit wurde ein aus der Biologie stammender Datensatz verwendet, der spezifische Attribute wie DNA, RNA, Zelllinien, Zelltypen und Proteine enthält. Dieser Datensatz ermöglicht es, NER unter realistischen Bedingungen, die typisch für die Biologie sind, zu testen.

Ein besonders relevanter Bereich für NER ist auch der E-commerce, da es hier ermöglicht wird, wichtige Informationen aus Produktbeschreibungen, Kundenbewertungen und anderen unstrukturierten Texten zu extrahieren. So können beispielsweise Marken, Produktnamen und Preise automatisch erkannt werden. Diese Informationen können dann verwendet werden, um Produkte besser zu kategorisieren, die Suchergebnisse zu verbessern und Personalisierte Angebote zu generieren. Durch die Automatisierung dieses Prozesses kann es das E-Commerce-Unternehmen einen Wettbewerbsvorteil verschaffen und die Kundenzufriedenheit erhöhen.

Es stellt eine große Chance dar, die Fähigkeiten im Bereich Maschinelles Lernen und der Textanalyse zu vertiefen und anwenden zu können und bringt somit auch einen wichtigen Beitrag zur Zukunft der Textanalyse-Technologie.

\section{Problem und Zielstellung}

Dadurch es in der Praxis häufig der Fall sein kann, dass Daten aufgrund von fehlenden Ressourcen oder Zeitnot künstlich annotiert werden, um sie für maschinelles Lernen verwenden zu können, kann dies dazu führen, dass die Ergebnisse ungenauer sind, als wenn die Annotation von Menschen durchgeführt wird. Dies liegt daran, dass automatisierte Annotationen oft nicht so zuverlässig sind wie die von Menschen durchgeführten Annotierungen.

Daher ist es besonders wichtig, in der Lage zu sein, NER mit schwach annotierten Daten erfolgreich durchzuführen. Dies erfordert oft die Verwendung von fortgeschrittenen Techniken und Methoden, um die Muster und Merkmale von Eigennamen in den Daten zu erkennen und zu extrahieren. 

Durch die erfolgreiche Durchführung von NER mit schwach annotierten Daten kann man das Problem der ungenauen Ergebnisse, die durch künstliche Annotation entstehen, minimieren und so die Genauigkeit und Leistung von maschinellen Lernmodellen verbessern. Es ermöglicht auch die Verwendung von größeren und vielfältigeren Datenmengen, die sonst nicht verfügbar wären, und erhöht die allgemeine Anwendbarkeit von NER in verschiedenen Anwendungsbereichen.

Das Problem bei NER mit schwach gekennzeichneten Daten besteht darin, dass die Daten nicht vollständig annottiert sind, d.h. dass nicht alle Eigennamen in einem Text richtig identifiziert und gekennzeichnet sind. Dies kann es schwieriger machen, für ein maschinelles Lernmodell die Muster und Merkmale von Eigennamen zu erlernen, was zu geringerer Leistung und Genauigkeit führen kann, wenn man versucht, Eigennamen in neuem Text zu identifizieren und extrahieren. Das kann auch zu mehr Fehlern im Erkennungsprozess führen, was besonders problematisch sein kann für Anwendungen, die auf genauer NER angewiesen sind, wie z.B. Informationsextraktion und natürliches Sprachverständnis.

In dieser Arbeit wird das Problem zunächst als Tokenklassifizierungsaufgabe definiert. Gegeben ist ein Kontext $T = (w_1^t, w_2^t, ..., w_m^t)$ und dessen Attribut $A = (w_1^a, w_2^a, ..., w_n^a)$, unser Ziel ist es, den Wert $V = (w_1^v, w_2^v, ..., w_e^v)$ zu generieren. Ein beispielhafter Kontext aus dem JNLPBA Datenset ist ``IL-2 gene expression and NF-kappa B activation through CD28 requires reactive oxygen production by 5-lipoxygenase.''. Betrachtet man die beiden Attribute ``expression'' und ``CD28''. Es soll der Wert ``NUL'' für das Attribut "expression", da dieses Attribut im Kontext nicht vorhanden ist und "Protein" für das Attribut "CD28" generiert werden. Folgend wird das Problem in dieser Arbeit als Sequenz zu Sequenz Problem definiert. In diesem Ansatz geht es darum, V als Stringausgabe zu generieren wobei der Kontext T als Eingabestring betrachtet wird. Dabei ist das Ziel nur die V zu erkennen welche nicht NULL sind.

\section{Aufbau der Arbeit }

TODO zusammen mit Inhaltsverzeichnis