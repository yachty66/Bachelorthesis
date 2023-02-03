\chapter{Methode}

Für die Forschungsfrage "Welche Methoden sind geeignet, um NER Probleme auf schwach annotierten Daten zu lösen?" wurden verschiedene Ansätze gewählt, um zu testen wie eine Erfolgreiche Methodik gestaltet werden könnte. Im folgenden Abschnitt werden die einzelnenen Vorgehensweisen genauer beschrieben.


\section{Datensatz}

Als Datensatz für diese Arbeit wurde JNLPBA (https://aclanthology.org/W04-1213.pdf) gewählt. 
Dieser Datensatz ist dem  GENIA version 3.02 corpus ensprungene. Dabei handelt es sich um einen Biomedizinischen Datensatz, bei dem 2000 Abstracts von medizinischen Papern mit Hand annotiert wurden. Für die Experimente wurde die auf Huggingface vorliegende Version des Datensatzes gewählt (https://huggingface.co/datasets/jnlpba). Diese Version enthält die in der Tabelle ~\ref{table:JNLPBA_full} dargestellten Attribute mit Ihrer jeweiligen Häufigkeit. Weiterhin wird in der Tabelle die durchschnittliche Wort Länge eines Satzes und die durchschnittliche Anzahl an Attributen, die in einem Satz vorkommen dargestellt. Die Angaben werden jeweils für den Trainings Datensatz und den Validierungsdatensatz aufgezeigt.


\begin{table}[htp]
\caption{Datensatz Statistik}
\centering
\begin{tabular}{llrc}
\toprule
\cmidrule(r){1-2}
& Metrik & Trainings Datensatz & Validierungs Datensatz \\
\midrule
& O & 205885 & 106348 \\
& B-DNA & 4878 & 1366 \\
& I-DNA & 8011 & 2365 \\
& B-RNA & 479 & 154 \\
& I-RNA & 715 & 244 \\
& B-cell_line & 2097 & 735 \\
& I-cell_line & 4124 & 1420 \\
& B-cell_type & 3718 & 2409 \\
& I-cell_type & 4791 & 3686 \\
& B-protein & 16897 & 6276 \\
& I-protein & 13148 & 5929 \\
\bottomrule
\end{tabular}
\label{table:JNLPBA_full}
\end{table}




\section{Token Klassifizierung}

Tokenklassifizierung kann als Klassfikationsproblem dargestellt werden bei dem jeder Token eines Textes einem bestimmten Attribut zugeschrieben wird. Beispielhaft könnte ein Set an verfügbaren Attributen für einen Text  Länder, Städte oder Flüsse sein. Die Aufgabe des Klassifikationsmodell besteht darin basierend auf den Merkmalen des Tokens Vorhersagen zu treffen welches Attribut dem Token zugeordnet werden soll. Mathemstisch wird dieses Problem wiefolgt beschrieben:

$$f: X -> Y$$

Dabei ist $X$ das Merkmal welches durch einen Token repräsentiert wird und $Y$ die Menge aller möglichen Klassen die für die Aufgabe zur Verfügung stehen. Das Ziel ist es eine Funktion $f$ zu finden, die die bestmögliche Vorhersage basierend auf den Merkmalen $X$ für jeden einzelnen Token erbringt.

Es gibt verschiedene Vorteile das Problem der NER mit einem Tokenklassifizierung Ansatz zu modellieren. Die Klassifizierung kann einen Text mit komplexer Struktur atomarer darstellen, indem spezifisch Tokens gewählt werden auf denen sich dann Aussagen treffen können. Die Vergabe eines bestimmten Attributes lässt sich dann genau auf den jeweiligen Token zurück führen. Dadurch die Klassifiezierung diretk auf Tokenbasis basis kann es auch einfacher sein Tokenklassifizierung durchzuführen, weil das Problem jedem Token einem Attribut zuzuordnen klar definiert ist. Die Ziele die NER verfolgt sind ähnlich zu dem Problem, welches Tokenklassifizirung löst. 



Der Grund für die Auswahl der herangehensweise ist der, dass 





\section{Token Klassifizierung mit schwachen Daten}
\section{Seq2seq mit starken Daten}
\section{Seq2seq mit schwachen Daten}
