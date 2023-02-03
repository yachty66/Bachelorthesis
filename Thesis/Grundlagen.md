\chapter{Grundlagen}

\section{Neuronale Netzwerke}

Neurale Netzwerke sind Algorithmen, die dafür entwickelt wurden, die Funktionen des menschlichen Gehirns nachzubilden. Sie basieren auf der Idee, dass das menschliche Gehirn aus vielen Neuronen besteht, die miteinander verbunden sind und so in der Lage sind, komplexe Aufgaben zu lösen. Der erste Algorithmus, der auf dieser Idee basierte, war das McCulloch-Pitts Neuron Model, entwickelt von Warren McCulloch und Walter Pitts im Jahr 1943 \cite{mcculloch_logical_nodate} ~\ref{fig:MP}.

\begin{figure}[htp]
\centering
\includegraphics{figures//MP_Neuron_deutsch.drawio.pdf}
  \caption{MP Neuron}
\label{fig:MP}
\end{figure}

Das McCulloch-Pitts Neuron Model besteht aus einem einzelnen Neuron. Es nimmt mehrere Eingabe-Signale entgegen und wandelt diese in ein Ausgangssignal um. Die Aktivierung des Neurons erfolgt durch die Anwendung einer bestimmten Funktion, die in der Formel dargestellt wird:

$$y = 1 \text{ wenn } w_1 \cdot x_1 + w_2 \cdot x_2 + \dots + w_n \cdot x_n \ge \theta$$
$$y = 0 \text{ sonst }$$

In dieser Formel ist $y$ die Ausgabe des Neurons, $x_n$ die Eingabe des Neurons, $w_n$ die Gewichtung der Eingaben und $\theta$ der Schwellenwert. Wenn die Summe der gewichteten Eingaben größer oder gleich dem Schwellenwert ist, wird das Neuron aktiviert und gibt eine 1 aus, andernfalls bleibt es inaktiv und gibt eine 0 aus.

Obwohl das McCulloch-Pitts Neuron Model ein wichtiger Schritt in der Entwicklung neuronaler Netzwerke war, hatte es einige Einschränkungen und konnte nur bestimmte Aufgaben lösen. Der nächste Schritt in der Geschichte neuronaler Netzwerke war die Entwicklung des Perzeptrons im Jahr 1957 durch Frank Rosenblatt \cite{rosenblatt_perceptron_1958}, welches es ermöglichte, komplexere Probleme zu lösen ~\ref{fig:Perzeptron}.

\begin{figure}[htp]
\centering
\includegraphics{figures/Perceptron_deutsch.drawio.pdf}
  \caption{Perzeptron }
  \label{fig:Perzeptron}
\end{figure}

Jedes Neuron im Perzeptron nimmt mehrere Eingabe-Signale entgegen und wandelt diese in ein Ausgangssignal um. Die Aktivierung des Neurons erfolgt durch die Anwendung einer bestimmten Funktion, die in der Formel dargestellt wird:

$$y = 1 \text{ wenn } w_1 \cdot x_1 + w_2 \cdot x_2 + \dots + w_n \cdot x_n \ge \theta$$
$$y = 0 \text{ sonst }$$

In dieser Formel ist $y$ die Ausgabe des Neurons, $x_i$ die Eingabe des Neurons, $w_i$ die Gewichtung der Eingaben und $\theta$ der Schwellenwert. Wenn die Summe der gewichteten Eingaben größer oder gleich dem Schwellenwert ist, wird das Neuron aktiviert und gibt eine 1 aus, andernfalls bleibt es inaktiv und gibt eine 0 aus. Die Funktion ist diesselbe wie die Funktion des McCulloch-Pitts Neuron Model, allerdings ist die Architektur des Perzeptrons darauf ausgelegt die Gewichte der Eingabe-Signale anzupassen. Das Perzeptron kann auch mit anderen Aktivierungsfunktionen ausgestattet werden. 

Die Entwicklung des mehrschichtigen Perzeptron (MLP) im Jahr 1986 durch Rumelhart, Hinton und Williams, war ein weiterer wichtiger Schritt in der Geschichte neuronaler Netzwerke. Im Gegensatz zum einfachen Perzeptron, das nur aus einer Schicht von Neuronen besteht, hat das MLP mehrere Schichten von Neuronen, wodurch es komplexere Probleme lösen kann. Das MLP besteht aus einer Eingabeschicht, einer oder mehreren versteckten Schichten und einer Ausgabeschicht. Jede Schicht besteht aus mehreren Neuronen, die miteinander verbunden sind und die Eingaben von der vorherigen Schicht erhalten ~\ref{fig:MLP}.

\begin{figure}[H]
\centering
\includegraphics{figures/MLP.drawio.pdf}
  \caption{Mehrschichtiges Perzeptron}
  \label{fig:MLP}
\end{figure}

Die Architektur des MLP ermöglicht es, nicht linear separierbare Probleme zu lösen und es hat sich als sehr erfolgreich in vielen Anwendungen gezeigt, wie zum Beispiel Bilderkennung, Sprachverarbeitung und natürliche Sprachverarbeitung.

Es kann auch mit anderen Aktivierungsfunktionen wie Sigmoid, ReLU, Tanh etc. ausgestattet werden und diese können die Leistung verbessern und ermöglichen es das Modell nicht nur lineare Probleme lösen kann. Ein weiteres wichtiges Merkmal des MLP ist das Lernverfahren, welches Backpropagation genannt wird und es ermöglicht das Modell an die gegebenen Daten anzupassen.

Es kommt darauf an wie viele Schichten ein MLP besitzt, aber eine allgemeine Formel für die Ausgabe eines Neurons in einer beliebigen Schicht des MLP kann jedoch wie folgt dargestellt werden:

$$y_i = f(w_{1,i} \cdot x_1 + w_{2,i} \cdot x_2 + \dots + w_{n,i} \cdot x_n + b_i)$$

In dieser Formel ist $y_i$ die Ausgabe des Neurons $i$ in Schicht $j$, $f$ die Aktivierungsfunktion, $x_i$ die Eingabe des Neurons aus der vorherigen Schicht, $w_{i,j}$ die Gewichtung der Eingaben des Neurons $i$ in Schicht $j$ und $b_i$ der Schwellenwert des Neurons $i$ in Schicht $j$. Die Ausgabe des Neurons $i$ in Schicht $j$ hängt also von den Eingaben und Gewichtungen der vorherigen Schicht sowie von der Aktivierungsfunktion des Neurons ab.

Das McCulloch-Pitts Neuron Model, das Perzeptron und das MLP sind alle unterschiedliche Modelle der künstlichen neuronalen Netze. Jedes dieser Modelle hat seine eigenen Eigenschaften und Anwendungen ~\ref{table:Vergleich}.

\begin{table}[htp]
\caption{Unterschiede zwischen McCulloch-Pitts Neuron-Modell, Perzeptron und Multilayer Perzeptron}
\centering
\begin{tabular}{llrc}
\toprule
\cmidrule(r){1-2}
& McCulloch-Pitts & Perzeptron & Multilayer Perzeptron \\
\midrule
& 1 & 1 & mehrere \\
& Ja/Nein & Ja/Nein & verschiedene \\
& 1 & beliebig & beliebig \\
& binäre Entscheidungen & lineare Trennung & komplexe Aufgaben \\
\bottomrule
\end{tabular}
\label{table:Vergleich}
\end{table}

\section{Backpropagation}

Backpropagation ist eine Methode, die in künstlichen neuronalen Netzwerken verwendet wird, um die Gewichte eines neuronalen Netzwerks anzupassen, um Vorhersagen basierend auf einem Datensatz zu liefern. Der Algorithmus betrachtet zuerst die Gewichte an der Ausgabeschicht und geht von dort aus bis hin zur Eingabeschicht ~\ref{fig:Backpropagtion}.

\begin{figure}[htp]
\centering
\includegraphics{figures/Backpropagation.pdf}
  \caption{Backpropagation }
  \label{fig:Backpropagtion}
\end{figure}

Angenommen, es gibt zwei Neuronen $x_1$ und $x_2$ in der Eingabeschicht, zwei Neuronen $h_1$ und $h_2$ in der versteckten Schicht, $y$ als Ausgangsneuron und die Gewichte $w1, w2, w3, w4, w5$ und $w6$ zwischen den Neuronen.

Die Ausgabe $\hat{y}$ wird folgendermaßen berechnet:

$$\hat{y} = f(w1 * x1 + w2 * x2 + w3 * h1 + w4 * h2)$$

$f$ ist die Aktivierungsfunktion.

Um die Werte der einzelnen Gewichte so anzupassen, dass genauere Ergebnisse erhalten werden, verwendet man Backpropagation. Bei einem Durchlauf kommt es zu dem Ergebnis $\hat{y}$. Das ist die tatsächliche Ausgabe eines neuronalen Netzwerks nach einem Durchlauf der Daten. Die tatsächliche Ausgabe wird mit dem Wert der erwarteten Ausgabe verglichen. Man berechnet den Fehler $\delta$ dieser zwei Werte.

$$\delta = (y - \hat{y})^2$$

Für den Fehler kann man verschiedene mathematische Funktionen verwenden. Eine übliche Funktion ist die der mittleren quadratischen Abweichung (MSE).

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y_i})^2$$

Die Funktion quadriert jedes einzelne Fehlersignal und summiert am Ende alles zusammen und teilt es durch die Anzahl der Datenpunkte. Das Ergebnis ist der durchschnittliche Fehler aller quadrierten Datenpunkte. Es wird quadriert, weil die Ergebnisse sich dadurch leichter weiterverarbeiten lassen.

Nachdem das Fehlersignal berechnet wurde, besteht der nächste Schritt darin, diesen Fehler rückwärts durch das Netzwerk zu propagieren, um damit die Gewichte $w1, w2, w3, w4, w5$ und $w6$ zu erneuern.

Zuerst berechnet man den Gradient $\nabla$ der Verlustfunkion mit Respekt zu jedem einzelnen Gewicht im Netzwerk. Das passiert mit der Kettenregel der Infinitesimalrechnung. Dies ermöglicht die Berechnung der Ableitung der Verlustfunktion in Bezug auf das Gewicht.

Mit der Verlustfunktion $MSE$ würde der Gradient für das erste Gewicht folgendermaßen definiert werden:

$$\frac{\partial MSE}{\partial w1} = \frac{2}{n}\sum_{i=1}^{n} (\hat{y_i} - y_i) \frac{\partial \hat{y_i}}{\partial w1}$$

Die Berechnung des Gradienten ermöglicht es, die Gewichte in Richtung des negativen Gradienten zu aktualisieren, um den Fehler zu minimieren. Dieser Prozess wiederholt sich für jedes Gewicht im Netzwerk, bis das Netzwerk eine akzeptable Genauigkeit erreicht hat.

Backpropagation ist ein wichtiger Algorithmus in künstlichen neuronalen Netzwerken und ermöglicht es, das Netzwerk auf einem gegebenen Datensatz zu trainieren und somit Vorhersagen zu erstellen.

\section{Tiefes Lernen}

Tiefes Lernen ist die Weiterentwicklung der vorher beschriebenen einfachen neuronalen Netzwerke und ein Teilbereich des maschinellen Lernens ~\ref{fig:tiefes_lernen}.

\begin{figure}[htp]
\centering
\includegraphics{figures/Hierarchie.pdf}
\caption{Einordnung Tiefes Lernen }
\label{fig:tiefes_lernen}
\end{figure}

Tiefes Lernen wurde durch die 2010er Jahre bekannt. Zu dieser Zeit kam es zur Verfügbarkeit von großen Datensätzen und Fortschritten in der Rechenleistung. Im Jahr 2012 erzielte ein Tiefes Lernen-Modell, das von Google entwickelt wurde \cite{krizhevsky_imagenet_2012}, Durchbrüche in der Bilderkennung und übertraf alle vorherigen Modelle auf dem ImageNet-Datenset. Diese Leistung förderte ein breiteres Interesse an Tiefem Lernen und führte zu Fortschritten in verschiedenen Anwendungsbereichen. Tiefes Lernen-Algorithmen können unter anderem für Bild- und Spracherkennung, NLP und das Spielen von Spielen wie Schach und Go verwendet werden. Es gibt verschiedene Architekturen, die für verschiedene Arten des Lernens von Daten eingesetzt werden.

Anstatt einer Schicht, die die Eingabewerte von Neuronen verarbeitet, gibt es bei einem Tiefes Lernen-Modell mehrere Schichten. Die genaue Anzahl der Layer, die benötigt werden, um ein Modell zu einem Tiefen Lernen-Modell zu machen, ist nicht definiert. Der Begriff Tiefes Lernen ist daher ein eher allgemeiner Begriff, der Modelle beschreibt, die mehrere Schichten enthalten. Eine Schicht besteht jeweils aus Eingabewerten, einer Funktion, die die Eingabewerte verarbeitet und Ausgabewerte erzeugt.

\section{NLP}

Natural Language Processing (NLP) ist ein Teilbereich der Linguistik und des maschinellen Lernens. Die Verarbeitung menschlicher Sprache findet in vielen Bereichen Anwendung, wie zum Beispiel Eigennamen-Erkennung, maschinelle Übersetzung, Spracherkennung und Sentiment-Analyse. Da die sinnvolle Weiterverarbeitung von Sprache eine komplexe Aufgabe ist, da Wörter in verschiedenen Kontexten unterschiedliche Bedeutungen haben können, muss ein tiefes Lernen Modell mit vielen Daten trainiert werden, um die verschiedenen Kontexte zu erkennen.

Ein Beispiel für unterschiedliche Wortbedeutungen in unterschiedlichen Zusammenhängen:

``Neben der Kirche befindet sich eine Bank.'' (Hier ist ``Bank'' ein Sitzplatz und nicht ein Finanzinstitut)

Sprache muss zunächst auf ein für den Computer verständliches Format reduziert werden, um damit Algorithmen entwickeln zu können.

\subsection{Tokenisierung}

Tokenisierung ist ein wichtiger Schritt im NLP und ist oft der erste Schritt in einer NLP-Pipeline. Sie dient dazu, Text in kleinere Einheiten aufzuteilen, um die Analyse und Weiterverarbeitung des Textes zu vereinfachen. Mit kleineren Einheiten wird es einfacher, Muster zu erkennen, Bedeutungen zu extrahieren oder andere Operationen durchzuführen, die man mit dem Text ausführen möchte.

Es gibt verschiedene Arten der Tokenisierung, wie zum Beispiel Wort-Tokenisierung, Satz-Tokenisierung und Tokenisierung von einzelnen Satzzeichen. Welche Methode am besten geeignet ist, hängt von der jeweiligen Aufgabe ab. Wort-Tokenisierung eignet sich beispielsweise für Aufgaben wie Textklassifizierung oder Übersetzung, während die Tokenisierung von Satzzeichen für die Erkennung von handgeschriebener Schrift besser geeignet sein kann. Beispiele für ``Der Hund bellt.'':
    \begin{itemize}
        \item Wort-Tokenisierung: \textquotesingle Der \textquotesingle, \textquotesingle Hund \textquotesingle, \textquotesingle bellt.\textquotesingle
        \item Satz-Tokenisierung: \textquotesingle Der Hund bellt.\textquotesingle
        \item Satzzeichen-Tokenisierung: \textquotesingle D \textquotesingle,\textquotesingle e \textquotesingle, \textquotesingle r \textquotesingle, \textquotesingle H \textquotesingle, \textquotesingle u \textquotesingle, \textquotesingle n \textquotesingle, \textquotesingle d \textquotesingle, \textquotesingle b \textquotesingle, \textquotesingle e \textquotesingle, \textquotesingle l \textquotesingle, \textquotesingle l \textquotesingle, \textquotesingle t \textquotesingle, \textquotesingle . \textquotesingle
    \end{itemize}
    
\subsection{Vektorisierung}

Vektorisierung beschreibt den Prozess, Text in numerische Vektoren umzuwandeln. Diese Umwandlung kann dann als Input für Algorithmen des maschinellen Lernens verwendet werden. Es gibt verschiedene Methoden zur Vektorisierung von Daten, die davon abhängen, in welchem Kontext man die Vektorisierung verwenden möchte.

One-Hot-Codierung ist eine Methode, bei der jedes Wort durch einen binären Vektor dargestellt wird. Dieser Vektor enthält eine ``$1$'' an der Position, die dem Wort entspricht, und sonst ``$0$''. Diese Methode kann jedoch zu einem großen Vokabular führen, das die Anzahl der einzigartigen Worte in einem Text darstellt. Beispiel für ``Der Hund bellt.'':

\begin{align*}
    \text{Der} &= \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix} \\
    \text{Hund} &= \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix} \\
    \text{bellt.} &= \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix} 
\end{align*}

Worteinbettungen werden auf vielen Textdaten trainiert. Die Vektoren für jedes Wort werden so gelernt, dass der Vektor den Kontext widerspiegelt, in dem das Wort auftritt. Wörter, die in ähnlichen Kontexten auftreten, tendieren dazu, ähnliche Vektoren zu haben, während Wörter mit unterschiedlichen Kontexten unterschiedlichere Vektoren haben ~\ref{fig:Worteinbettung}.

\begin{figure}[H]
\centering
\includegraphics{figures/Worteinbettung.pdf}
\caption{Worteinbettung }
\label{fig:Worteinbettung}
\end{figure}

\section{NER}

Named-Entity-Recognition (NER) oder Eigennamen Erkennung im deutschen ist eine Aufgabe im Bereich des NLP, bei der Eigennamen wie Personen, Organisationen und Orte in einem Text erkannt werden. Dies macht NER nützlich für Anwendungen wie Fragebeantwortung, Informationsextraktion und die Zusammenfassung von Dokumenten ~\ref{fig:NER}.

\begin{figure}[H]
\centering
\includegraphics{figures/NER.png}
\caption{Beispiel NER }
\label{fig:NER}
\end{figure}

Es gibt verschiedene Ansätze zur Durchführung von NER, wie regelbasierte Systeme, maschinelles Lernen und hybride Systeme. Eine Methode, die häufig in Kombination mit NER verwendet wird, ist die Wortartentagging. Dabei wird jedem Wort im Text seine grammatische Rolle zugeordnet, wie beispielsweise Verb, Adjektiv oder Subjekt. Beispielsweise ergibt das Wortartentagging für den Satz ``Der Hund bellte in Venedig.'' die Ausgabe: ``Der'' (Artikel), ``Hund'' (Substantiv), ``bellte'' (Verb), ``in'' (Präposition) und  ``Venedig'' (Substantiv). Da Eigennamen oft auch als Substantive auftreten, kann das Wortartentagging dazu beitragen, Eigennamen genauer zu identifizieren.

Eine weitere Methode, die in Kombination mit NER verwendet wird, ist die Abhängigkeitsanalyse. Hierbei wird die grammatikalische Struktur eines Satzes untersucht und die Abhängigkeiten zwischen den Wörtern bestimmt. Diese Abhängigkeiten werden in einem Abhängigkeitsanalysebaum dargestellt, bei dem jedes Wort als Knoten repräsentiert wird und die Abhängigkeiten als Kanten zwischen den Knoten. Beispielsweise ergibt die Abhängigkeitsanalyse für den Satz ``Der Hund bellte in Venedig.'' den Baum ~\ref{fig:Abhängigkeitsanalysebaum}:

\begin{figure}[H]
\centering
\Tree [.bellte [.Hund Der ] [.in [.Venedig ] ] ]
\caption{Beispiel Abhängigkeitsanalysebaum}
\label{fig:Abhängigkeitsanalysebaum}
\end{figure}

\section{Annotation}

Annotation, beschreibt den Prozess, bei dem Datenpunkten eine bestimmte Klasse zugeordnet werden. Annotationen dienen dazu, einen Datenpunkt eindeutig einer bestimmten Klasse zuzuordnen. Zum Beispiel wird bei NER jedem Wort Annotationen zugeordnet, die Aussagen über das jeweilige Attribut des Wortes treffen. Annotationen werden verwendet, um ein maschinelles Lernmodell zu trainieren, um Vorhersagen für neue, unbekannte Daten treffen zu können.

Es gibt verschiedene Methoden zum annotieren von Daten. Eine Methode ist das manuelle Annotieren durch Menschen. Ein Beispiel dafür ist das ImageNet-Datenset, das teilweise von Menschen annotiert wurde \cite{deng_imagenet_2009}. Der Vorteil dieser Methode ist, dass eine hohe Qualität der Annotationen erwartet werden kann, allerdings ist sie im Vergleich zu anderen Techniken sehr zeitaufwendig.

Eine andere Methode ist das automatische annotieren, beispielsweise mittels regelbasierter Systeme oder aktiv lernender Algorithmen. Der Vorteil dieser Methode ist, dass das annotieren schneller durchgeführt werden kann, jedoch ist die Qualität der Annotationen nicht so hoch wie beim manuellen annotieren.

\subsection{Regelbasierte Systeme}

Eine Möglichkeit des automatischen annotieren ist die Verwendung von regelbasierten Systemen. Diese Methode beginnt mit der manuellen Erstellung eines Datensatzes mit Annotationen, beispielsweise könnte der Satz ``Angela Merkel wurde 1954 in Hamburg geboren.'' die annotierten Eigennamen ``Angela Merkel'' und ``Hamburg'' enthalten. Der Datensatz wird dann mit einem überwachten Algorithmus für maschinelles Lernen trainiert, dessen Ziel es ist, die Muster der Entitäten im Text zu erkennen.
Das Model kann als eine Funktion $f$ dargestellt werden, die eine Eingabe $x$ (ein Textdokument) entgegennimmt und eine Ausgabe $y$ (die Labels für die bestimmten Eigennamen) erzeugt. Die Funktion $f$ lernt mit einem Trainingsalgorithmus aus dem annotierten Datensatz.
Wenn das Model trainiert ist, kann es neue Annotaionen für eingebene Textdokumente mit Eigennamen bestimmen. Beispielsweise sollte das Model bei der Eingabe von ``Wolfgang Schäuble wurde 1942 in Freiburg geboren.'' die Eigennamen ``Wolfgang Schäuble'' und ``Freiburg'' vorhersagen.
Die erstellten Annotationen werden verwendet, um neue Regeln zu erzeugen oder bestehende zu erweitern.

\subsection{Aktiv lernende Algorithmen}

Eine weitere Möglichkeit des automatischen annotierens ist die Verwendung von aktiv lernenden Algorithmen. Diese Methode beginnt ebenfalls mit einem kleinen annotierten Datensatz, das mit einem überwachten Algorithmus für maschinelles Lernen trainiert wird. Das trainierte Model wird dann verwendet, um Vorhersagen für nicht annotierte Daten zu treffen. Anschließend werden die Beispiele ausgewählt, bei denen das Model die größte Unsicherheit bei der Vorhersage hat, beispielsweise Eigennamen, die im Model nicht vorkommen. Diese ausgewählten Beispiele werden von Menschen annotiert und dem annotierten Datensatz hinzugefügt. Das Model wird anschließend erneut trainiert. Dieser Schritt wird so oft wiederholt, bis die gewünschte Leistung erreicht ist. Es ist auch möglich, die Annotationen zufällig auszuwählen, anstatt die Annotationen zu wählen, bei denen das Model die größte Unsicherheit hat. Der Nachteil beim zufälligen Auswählen ist jedoch, dass das Model länger benötigt, um bessere Leistungen zu erzielen.

Es ist wichtig zu beachten, dass die Qualität der Annotationen für alle Ansätze von großer Bedeutung ist, um schlechte Trainingsergebnisse zu vermeiden und einen Algorithmus effektiv lernen zu lassen. Insbesondere bei automatischen Annotations-Methoden, wie Regelbasierten Systemen und aktiven lernenden Algorithmen, ist es wichtig, die Qualität der Annotationen sorgfältig zu überwachen und gegebenenfalls manuelle Korrekturen vorzunehmen.

\section{RNN}

Ein rekurrentes neuronales Netzwerk (RNN) ist ein Typ von neuronalem Netzwerk, das dafür entwickelt wurde, aufeinander folgende Daten zu verarbeiten.

In RNNs wird jeder Zeitpunkt in der Eingabe vom Netzwerk verarbeitet. Beispielsweise bei der Eingabe eines Satzes wie ``Wie spät ist es?'', ist jedes einzelne Wort gleich einem Zeitpunkt. Die Ausgabe zu jedem Zeitpunkt hängt von der Ausgabe des vorherigen Zeitpunkts ab. Das ermöglicht dem Netzwerk, Abhängigkeiten, die sich über mehrere Zeitpunkte erstrecken, zu erkennen ~\ref{fig:RNN}.

Angenommen, man hat ein Modell für Übersetzung. Wenn das RNN den Satz ``Wie spät ist es?'' für eine Übersetzung bearbeiten soll, muss man den Satz zunächst in ein sequentielles Datenformat bringen, zum Beispiel ``Wie'', ``spät'', ``ist'', ``es'', ``?''. Dann wird dem RNN zuerst ``Wie'' als Eingabe übergeben, danach ``spät'' mit der vorherigen Eingabe, also ``Wie'' und so weiter, bis die Sequenz beendet ist. Die so entstehende Repräsentation kann dann beispielsweise in ein Feed-Forward-Netzwerk weiter verarbeitet werden ~\ref{fig:RNN.

\begin{figure}[H]
\centering
\includegraphics{figures/RNN_Beispiel_deutsch.pdf}
    \caption{Beispiel RNN }
    \label{fig:RNN}
\end{figure}

Mathematisch kann ein RNN wie folgt beschrieben werden:

$$h(t) = f(h(t-1), x(t))$$

wobei $h(t)$ die Ausgabe zu jedem Zeitpunkt $t$ ist, $h(t-1)$ die Ausgabe des vorherigen Zeitpunktes ist und $x(t)$ die Eingabe zu Zeitpunkt $t$ ist. Die Funktion $f$ repräsentiert das Innere eines RNN ~\ref{fig:RNN_Mathe}.

\begin{figure}[H]
\centering
\includegraphics{figures/RNN_mit_Mathe.pdf}
    \caption{Beispiel RNN }
    \label{fig:RNN_Mathe}
\end{figure}

\section{LSTM}

Long Short-Term Memory (LSTM) Netzwerke sind entwickelt worden, um das Problem der ``verschwindenden Gradienten'' in der Trainierung von traditionellen rekurrenten neuronalen Netzen (RNNs) zu lösen.

Traditionelle RNNs erneuern die verborgenen Zustände zu jedem Zeitpunkt, indem der vorherige verborgene Zustand und die aktuelle Eingabe verwendet werden. Dies kann jedoch Probleme beim Lernen von langfristigen Abhängigkeiten verursachen, da der Gradient des Fehlers in Bezug auf den verborgenen Zustand und die Gewichte tendenziell mit fortschreitenden Zeitpunkten verschwindet. Dies wird als das Problem der ``verschwindenden Gradienten'' bezeichnet.

LSTMs wurden entwickelt, um dieses Problem zu lösen, indem weitere Netzwerkstrukturen eingeführt werden, die als ``Gedächtniszellen'' und  ``Tore'' bezeichnet werden. Die Tore ermöglichen es den Gedächtniszellen, längerfristige Informationen zu speichern und abzurufen. Dadurch sind LSTMs in der Lage, langfristige Abhängigkeiten in den Daten zu behalten und die Leistung von RNNs zu übertreffen ~\ref{fig:lstm_zelle}.

\begin{figure}[H]
\centering
\includegraphics{figures/LSTM_cell.pdf}
\caption{Zelle eines LSTM \cite{lstm_zelle}}
\label{fig:lstm_zelle}
\end{figure}

An jedem Zeitpunkt $t$ bekommt das LSTM als Eingabe die aktuelle Eingabe $x_t$ und den vorherigen verborgenen Zustand $h_t-1$ und erzeugt einen neuen verborgenen Zustand $h_t$ und eine Ausgabe $y_t$. Der verborgene Zustand $h_t$ ist eine Funktion der aktuellen Eingabe $x_t$, dem vorherigen verborgenen Zustand $h_t-1$ und dem vorherigen Zellenzustand $c_t-1$. Der Zellenzustand $c_t$ ist eine ``Erinnerung'' an vergangene Eingaben und verborgene Zustände, die über die Zeit aufrechterhalten werden.

\section{Transformer}

Die Transformer-Architektur wurde in dem ``Paper Attention is All You Need'' von Vaswani et al. im Jahr 2017 vorgestellt \cite{vaswani_attention_2017}. Sie stellt eine verbesserte Alternative zu bestehenden Modellen wie LSTMs und RNNs dar, die sequentielle Daten modellieren.

Eine der wichtigsten Innovationen der Transformer-Architektur ist die Verwendung von Selbstaufmerksamkeitsmechanismen. Dadurch kann das Modell die Eingabedaten an jeder Stelle dynamisch bewerten und den Kontext von anderen Positionen bei der Durchführung von Vorhersagen nutzen. Im Gegensatz zu traditionellen Modellen werden hierbei keine Fenster mit festen Längen oder wiederkehrende Verbindungen verwendet, um den Kontext zu erfassen.

Die Transformer-Architektur besteht aus folgenden Hauptkomponenten: Selbstaufmerksamkeitsschichten, die es dem Modell ermöglichen, die Eingabedaten an jeder Position dynamisch zu gewichten und den Kontext von anderen Positionen zu nutzen, wenn Vorhersagen getroffen werden; Positionsbezogene Feed-Forward-Schichten, welche die Eingabedaten an jeder Position unabhängig verarbeiten und das Erkennen von komplexeren Mustern ermöglichen; einem Encoder, der die Sequenz der Eingabedaten aufnimmt und eine kontinuierliche Darstellung der Eingabe erzeugt; und einem Decoder, der die kontinuierliche Darstellung aufnimmt und eine Ausgabesequenz erzeugt. Der Encoder und der Decoder bestehen dabei aus mehreren Positionsbezogenen Feed-Forward-Schichten und Selbstaufmerksamkeitsschichten ~\ref{fig:Transformer}.

\begin{figure}[H]
\centering
\includegraphics{figures/transformer.pdf}
\caption{Transformer Architektur }
\label{fig:Transformer}
\end{figure}

\subsection{Selbstaufmerksamkeitsschichten}

Wenn $X$ die Eingabe-Daten sind, mit $X \in \mathbb{R}^{n \times d}$ wo $n$ die Länge der Sequenz und $d$ Dimension der Eingabedaten ist.
Seien $Q$, $K$ und $V$ Matrizen, die verwendet werden, um die Eingabedaten in verschiedene Räume zu projizieren, mit $Q \in \mathbb{R}^{n \times d_k}$, $K \in \mathbb{R}^{n \times d_k}$, und $V \in \mathbb{R}^{n \times d_v}$. Die Werte $d_k$ und $d_v$ sind die Dimensionen von den projizierten Räumen.

Die Selbstaufmerksamkeitsschichten berechnen die Aufmerksamkeitsgewichte für jede Position, indem sie das Skalarprodukt der projizierten Eingabedaten nimmt und durch die Quadratwurzel der projizierten Raumdimensionen dividiert:

$$ Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V $$

Die Ausgabe der Selbstaufmerksamkeitsschicht ist dann die gewichtete Summe der Eingabedaten gemäß den berechneten Aufmerksamkeitsgewichten.

\subsection{Positionsbezogene Feed-Forward-Schichten}

Wenn $X$ die Eingabe-Daten sind, mit $X \in \mathbb{R}^{n \times d}$ wo $n$ die Länge der Sequenz und $d$ Dimension der Eingabedaten ist. Seien $W_1 \in \mathbb{R}^{d \times d_h}$ und $W_2 \in \mathbb{R}^{d_h \times d}$ die Gewichtsmatrizen der ersten und zweiten vollständig verbundenen Schicht und $b_1 \in \mathbb{R}^{d_h}$ und $b_2 \in \mathbb{R}^{d}$ die Bias-Vektoren. $d_h$ ist die Anzahl der Neuronen in den vollständig verbundenen Schichten und $f(x)$ ist die Aktivierungsfunktion.

Die Positionsbezogenen Feed-Forward-Schichten berechnen die Ausgabe $Y \in \mathbb{R}^{n \times d}$ folgendermaßen:

$$ Y = f(X W_1 + b_1) W_2 + b_2 $$

Da jede Position der Eingabedaten unabhängig von anderen Positionen verarbeitet wird, ermöglicht diese Schicht das Erkennen von komplexeren Mustern in den Eingabedaten.

\section{Seq2seq}

Die seq2seq-Architektur ist ein häufig verwendetes Modell im maschinellen Lernen, das für Aufgaben wie maschinelle Übersetzung, Textzusammenfassung und Chatbot-Design verwendet wird.
Die Architektur des seq2seq-Modells besteht aus zwei Hauptkomponenten, dem Encoder und dem Decoder. Der Encoder erhält als Eingabe beispielsweise einen Satz und wandelt ihn in einen Kontextvektor um, der immer die gleiche Länge besitzt, unabhängig von der Länge der Eingabedaten. Der Kontextvektor wird dann an den Decoder weitergegeben, der für die Erzeugung der Ausgabe verantwortlich ist ~\ref{fig:Seq2seq}.

\begin{figure}[H]
\centering
\includegraphics{figures/Seq2seq.png}
\caption{Seq2seq }
\label{fig:Seq2seq}
\end{figure}

Der Encoder verarbeitet die Eingabedaten elementweise. Im Falle eines seq2seq-Modells für Übersetzungen wird zunächst das Wort  ``Der'' verarbeitet, dann  ``Hund'' usw. An jedem Zeitpunkt $t$ nimmt der Encoder eine Eingabe $x_t$ und verwendet sie, um den verborgenen Zustand $h_t$ mit dieser Eingabe zu erweitern. Der verborgene Zustand $h_t$ ist die Zusammenfassung aller bisher verarbeiteten Informationen. Mathematisch ist der verborgene Zustand wie folgt definiert:

$$h_t = f(h_{t-1}, x_t)$$

$f$ ist eine nichtlineare Aktivierungsfunktion wie die Sigmoid- oder Tanh-Funktion.

Wenn der Encoder die gesamte Eingabe verarbeitet hat, also den ganzen Satz ``Der Hund bellte'', erzeugt der Encoder einen finalen verborgenen Zustand $h_T$. $h_T$ wird als Kontextvektor $c$ verwendet und an den Decoder weitergegeben. Der Decoder verwendet den Kontextvektor, um den Ausgabesatz ``The dog barked'' zu produzieren.

Das Verarbeiten des Kontextvektors erfolgt beim Decoder ebenfalls elementweise. Das bedeutet, dass bei der Ausgabe  ``Der Hund bellte'' der Decoder zuerst  ``The'' erzeugt. An jedem Zeitpunkt $t$ nimmt der Decoder den vorherigen verborgenen Zustand $h_{t-1}$ und die vorherige Ausgabe $y_{t-1}$ und verwendet diese, um die aktuelle Ausgabe $y_t$ zu erzeugen. Mathematisch ist die Ausgabe an jedem Zeitpunkt wie folgt berechnet:

$$y_t = g(h_{t-1}, y_{t-1})$$

$g$ ist eine nichtlineare Aktivierungsfunktion wie die Sigmoid- oder Tanh-Funktion.

Der Decoder erzeugt Ausgaben $y_t$, bis die endgültige Zielausgabe ``Der Hund bellte'' erzeugt wurde.

Das gesamte seq2seq-Modell ist trainiert, um den Unterschied zwischen der vorhergesagten Ausgabe und der gewünschten Ausgabe zu minimieren. Dabei wird ein Optimierungsalgorithmus wie der stochastische Gradientenabstieg verwendet.

\section{Fine-Tuning}

Fine-Tuning ist eine Methode der maschinellen Lernung, bei der ein bereits trainiertes Modell auf einen neuen Datensatz angepasst wird, um die Leistung auf diesem spezifischen Datensatz zu verbessern. Dies wird häufig verwendet, wenn der verfügbare Datensatz für das ursprüngliche Training des Modells zu klein ist, um eine gute Leistung zu erzielen.

Dabei werden die Gewichte eines bereits trainierten Modells an einen neuen Datensatz angepasst. Dies erfolgt normalerweise durch die Durchführung weiterer Trainingsiterationen mit dem neuen Datensatz, während die Gewichte des Modells aktualisiert werden, um die neuen Daten besser zu passen.

Die Art und Weise, wie die Gewichte des Modells angepasst werden, hängt von der Architektur des Modells ab. In der Regel werden bei der Anpassung der Gewichte der höheren Schichten des Modells begonnen und dann nach unten vorgearbeitet.

In einem konventionellen neuronalen Netzwerk, das aus mehreren Schichten von Neuronen besteht, werden häufig nur die Gewichte der obersten Schichten angepasst, während die Gewichte der unteren Schichten statisch. Dies wird getan, um die Kenntnisse, die das Modell während des ursprünglichen Trainings erworben hat, beizubehalten ~\ref{fig:Fine-Tuning}. 

\begin{figure}[H]
\centering
\includegraphics{figures/Finetuning.pdf}
\caption{Fine-Tuning }
\label{fig:Fine-Tuning}
\end{figure}

Ein Beispiel für den Einsatz von Fine-Tuning wäre die Anpassung eines bereits trainierten Bilderkennungsmodells an eine neue Domäne, wie zum Beispiel medizinische Bilder. Das Modell wurde ursprünglich auf einen allgemeineren Datensatz trainiert, aber durch Fine-Tuning auf medizinische Bilder wird die Leistung auf diesem spezifischen Datensatz verbessert.

Ein weiteres Beispiel wäre die Anpassung eines bereits trainierten NLP-Modells an eine bestimmte Branche, wie zum Beispiel die Finanzbranche. Das Modell wurde ursprünglich auf allgemeine Texte trainiert, aber durch Fine-Tuning auf Finanztexte wird die Leistung auf diesem spezifischen Datensatz verbessert.

Fine-Tuning hat den Vorteil, dass es schneller und effizienter ist als das komplette Neutrainieren eines Modells, da es auf den bereits gelernten Kenntnissen aufbaut. Es ist jedoch wichtig zu beachten, dass der neue Datensatz für das Fine-Tuning relevant und repräsentativ sein sollte, um eine signifikante Verbesserung der Leistung zu erzielen.

In der Praxis wird Fine-Tuning häufig in Verbindung mit Transferlernen verwendet, bei dem das Modell auf ähnliche Aufgaben oder Domänen angewendet wird.

Fine-Tuning und Transferlernen sind beide Techniken der maschinellen Lernung, die darauf abzielen, die Leistung eines Modells auf einen neuen Datensatz oder eine neue Aufgabe zu verbessern, indem auf bereits erworbenen Kenntnissen aufgebaut wird. Der Unterschied zwischen den beiden Techniken liegt jedoch darin, wie dies erreicht wird.

Der Unterschied ist, dass Fine-Tuning sich auf die Anpassung der Gewichte eines bereits trainierten Modells an einen neuen Datensatz, indem weitere Trainingsiterationen durchgeführt werden, bezieht. Transferlernen bezieht sich dagegen auf die Verwendung von Kenntnissen, die das Modell während des Trainings auf einer bestimmten Aufgabe erworben hat, um die Leistung auf einer Zielaufgabe zu verbessern. Dies kann durch die Verwendung von Teilen des Modells (z.B. den Gewichten) oder durch die Verwendung von Kenntnissen, die das Modell auf der Aufgabe erworben hat, um die Leistung auf der Zielaufgabe zu verbessern, erfolgen.

