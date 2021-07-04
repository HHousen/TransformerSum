.. _convert_to_extractive:

Convert Abstractive to Extractive Dataset
=========================================

.. _convert_to_extractive_overview:

Overview
--------

This script will reformat an abstractive summarization dataset to be used for extractive summarization by determining the best extractive summary that maximizes ROUGE scores. It can be used on a dataset composed of the following file structure: ``train.source``, ``train.target``, ``val.source``, ``val.target``, ``test.source``, and ``test.target`` where each file contains one example per line and the lines of every ``.train`` file correspond to the lines in the respective ``.target``. All the datasets on the :ref:`extractive_supported_datasets` page will be processed to this format. You can also process any dataset contained in the `huggingface/nlp <https://github.com/huggingface/nlp>`_ library. If you use a dataset this way, the downloading and pre-processing will happen automatically.

.. _convert_to_extractive_option_1:

Option 1: Manual Data Download
------------------------------

Simply run ``convert_to_extractive.py`` with the path to the data. For example, with the :ref:`CNN/DM dataset <extractive_dataset_cnn_dm>`: ``python convert_to_extractive.py ./datasets/cnn_dailymail_processor/cnn_dm``. However, the recommended command is:

.. code-block:: bash

    python convert_to_extractive.py ./datasets/cnn_dailymail_processor/cnn_dm --shard_interval 5000 --compression --add_target_to test

* ``--shard_interval`` processes the file in chunks of ``5000`` and writes results to disk in chunks of ``5000`` (saves RAM)
* ``--compression`` compresses each output chunk with gzip (depending on the dataset reduces space usage requirement by about 1/2 to 1/3)
* ``--add_target_to`` will save the abstractive target text to the splits (in ``--split_names``) specified.

The default output directory is the input directory that was specified, but the output directory can be changed with ``--base_output_path`` if desired.

If your files are not ``train``, ``val``, and ``test``, then the ``--split_names`` argument will let you specify the correct naming pattern. The ``--source_ext`` and ``--target_ext`` let you specify the file extension of the source and target files respectively. These must be different so the process can tell each section apart.

.. _convert_to_extractive_option_2:

Option 2: Automatic pre-processing through ``nlp``
--------------------------------------------------

You will need to run the ``convert_to_extractive.py`` command with the ``--dataset``, ``--dataset_version``, ``--data_example_column``, and ``--data_summarized_column`` options set. To use the CNN/DM dataset you would set these arguments as shown below:

.. code-block::

    --dataset cnn_dailymail \
    --dataset_version 3.0.0 \
    --data_example_column article \
    --data_summarized_column highlights

View the help page (``python convert_to_extractive.py --help``) for more info about these options. The options are nearly identical to the :ref:`abstractive script <abstractive_supported_datasets>`.

.. important:: All of the :ref:`abstractive datasets <abstractive_supported_datasets>` can be converted for extractive summarization using this method.

The `live nlp viewer <https://huggingface.co/nlp/viewer>`_ visualizes the data and describes each dataset that can be used with through parameters.

Convert To Extractive Tips
--------------------------

**Large Dataset? Need to Resume?:** The ``--resume`` option will read the output directory and determine on which document the script left off based on the shard_file names. If ``--shard_interval`` was ``None`` then resuming is not possible. Resuming is guaranteed to produce the same output as if ``--resume`` was not used because of :meth:`convert_to_extractive.check_resume_success()`, which checks to make sure the last line in the shard file is the same as the line directly before the line to resume with.

**Speed: Running Slowly?** There is a ``--sentencizer`` option to detect sentence boundaries without parsing dependencies. Instead of loading a statistical model using ``spacy``, this option will initialize the ``English`` `Language <https://spacy.io/api/language#init>`_ object and add a ``sentencizer`` to the `pipeline <https://spacy.io/api/language#create_pipe>`_. This is much faster than a `DependencyParser <https://spacy.io/api/dependencyparser>`_ but is also less accurate since the ``sentencizer`` uses a simpler, rule-based strategy.

Custom Datasets
---------------

Any dataset in the format described in the :ref:`convert_to_extractive_overview` can be used with this script. Once converted, training should be the same as if using CNN/DM  from :ref:`Option 1 <convert_to_extractive_option_1>` because the :ref:`convert_to_extractive_api` script outputs a consistent format.

Extractive Dataset Format
^^^^^^^^^^^^^^^^^^^^^^^^^

This section briefly discusses the format of datasets created by the ``convert_to_extractive`` script.

The training and validation sets only need the ``src`` and ``labels`` keys saved as json. The ``src`` value should be a list of lists where each list contains a series of tokens (see below). The ``labels`` value is a list of 0s (not in summary) and 1s (sentence should be in summary) that is the same length as the ``src`` value (the number of sentences). Each value in this list corresponds to a sentence in ``src``. The testing set is special because it needs the ``src``, ``labels``, and ``tgt`` keys. The ``tgt`` key represents the target summary as a single string with a ``<q>`` between each sentence.

First document in **CNN/DM** extractive **training** set:

.. code-block::

    {'src': [['Editor', "'s", 'note', ':', 'In', 'our', 'Behind', 'the', 'Scenes', 'series', ',', 'CNN', 'correspondents', 'share', 'their', 'experiences', 'in', 'covering', 'news', 'and', 'analyze', 'the', 'stories', 'behind', 'the', 'events', '.'], ['Here', ',', 'Soledad', "O'Brien", 'takes', 'users', 'inside', 'a', 'jail', 'where', 'many', 'of', 'the', 'inmates', 'are', 'mentally', 'ill', '.'], ['An', 'inmate', 'housed', 'on', 'the', '"', 'forgotten', 'floor', ',', '"', 'where', 'many', 'mentally', 'ill', 'inmates', 'are', 'housed', 'in', 'Miami', 'before', 'trial', '.'], ['MIAMI', ',', 'Florida', '(', 'CNN', ')', '--', 'The', 'ninth', 'floor', 'of', 'the', 'Miami', '-', 'Dade', 'pretrial', 'detention', 'facility', 'is', 'dubbed', 'the', '"', 'forgotten', 'floor', '.', '"'], ['Here', ',', 'inmates', 'with', 'the', 'most', 'severe', 'mental', 'illnesses', 'are', 'incarcerated', 'until', 'they', "'re", 'ready', 'to', 'appear', 'in', 'court', '.'], ['Most', 'often', ',', 'they', 'face', 'drug', 'charges', 'or', 'charges', 'of', 'assaulting', 'an', 'officer', '--charges', 'that', 'Judge', 'Steven', 'Leifman', 'says', 'are', 'usually', '"', 'avoidable', 'felonies', '.', '"'], ['He', 'says', 'the', 'arrests', 'often', 'result', 'from', 'confrontations', 'with', 'police', '.'], ['Mentally', 'ill', 'people', 'often', 'wo', "n't", 'do', 'what', 'they', "'re", 'told', 'when', 'police', 'arrive', 'on', 'the', 'scene', '--', 'confrontation', 'seems', 'to', 'exacerbate', 'their', 'illness', 'and', 'they', 'become', 'more', 'paranoid', ',', 'delusional', ',', 'and', 'less', 'likely', 'to', 'follow', 'directions', ',', 'according', 'to', 'Leifman', '.'], ['So', ',', 'they', 'end', 'up', 'on', 'the', 'ninth', 'floor', 'severely', 'mentally', 'disturbed', ',', 'but', 'not', 'getting', 'any', 'real', 'help', 'because', 'they', "'re", 'in', 'jail', '.'], ['We', 'toured', 'the', 'jail', 'with', 'Leifman', '.'], ['He', 'is', 'well', 'known', 'in', 'Miami', 'as', 'an', 'advocate', 'for', 'justice', 'and', 'the', 'mentally', 'ill', '.'], ['Even', 'though', 'we', 'were', 'not', 'exactly', 'welcomed', 'with', 'open', 'arms', 'by', 'the', 'guards', ',', 'we', 'were', 'given', 'permission', 'to', 'shoot', 'videotape', 'and', 'tour', 'the', 'floor', '.', ' '], ['Go', 'inside', 'the', "'", 'forgotten', 'floor', "'", '»', '.'], ['At', 'first', ',', 'it', "'s", 'hard', 'to', 'determine', 'where', 'the', 'people', 'are', '.'], ['The', 'prisoners', 'are', 'wearing', 'sleeveless', 'robes', '.'], ['Imagine', 'cutting', 'holes', 'for', 'arms', 'and', 'feet', 'in', 'a', 'heavy', 'wool', 'sleeping', 'bag'], ['--', 'that', "'s", 'kind', 'of', 'what', 'they', 'look', 'like', '.'], ['They', "'re", 'designed', 'to', 'keep', 'the', 'mentally', 'ill', 'patients', 'from', 'injuring', 'themselves', '.'], ['That', "'s", 'also', 'why', 'they', 'have', 'no', 'shoes', ',', 'laces', 'or', 'mattresses', '.'], ['Leifman', 'says', 'about', 'one', '-', 'third', 'of', 'all', 'people', 'in', 'Miami', '-', 'Dade', 'county', 'jails', 'are', 'mentally', 'ill', '.'], ['So', ',', 'he', 'says', ',', 'the', 'sheer', 'volume', 'is', 'overwhelming', 'the', 'system', ',', 'and', 'the', 'result', 'is', 'what', 'we', 'see', 'on', 'the', 'ninth', 'floor', '.'], ['Of', 'course', ',', 'it', 'is', 'a', 'jail', ',', 'so', 'it', "'s", 'not', 'supposed', 'to', 'be', 'warm', 'and', 'comforting', ',', 'but'], ['the', 'lights', 'glare', ',', 'the', 'cells', 'are', 'tiny', 'and', 'it', "'s", 'loud', '.'], ['We', 'see', 'two', ',', 'sometimes', 'three', 'men', '--', 'sometimes', 'in', 'the', 'robes', ',', 'sometimes', 'naked', ',', 'lying', 'or', 'sitting', 'in', 'their', 'cells', '.'], ['"', 'I', 'am', 'the', 'son', 'of', 'the', 'president', '.'], ['You', 'need', 'to', 'get', 'me', 'out', 'of', 'here', '!', '"'], ['one', 'man', 'shouts', 'at', 'me', '.'], ['He', 'is', 'absolutely', 'serious', ',', 'convinced', 'that', 'help', 'is', 'on', 'the', 'way', '--', 'if', 'only', 'he', 'could', 'reach', 'the', 'White', 'House', '.'], ['Leifman', 'tells', 'me', 'that', 'these', 'prisoner', '-', 'patients', 'will', 'often', 'circulate', 'through', 'the', 'system', ',', 'occasionally', 'stabilizing', 'in', 'a', 'mental', 'hospital', ',', 'only', 'to', 'return', 'to', 'jail', 'to', 'face', 'their', 'charges', '.'], ['It', "'s", 'brutally', 'unjust', ',', 'in', 'his', 'mind', ',', 'and', 'he', 'has', 'become', 'a', 'strong', 'advocate', 'for', 'changing', 'things', 'in', 'Miami', '.'], ['Over', 'a', 'meal', 'later', ',', 'we', 'talk', 'about', 'how', 'things', 'got', 'this', 'way', 'for', 'mental', 'patients', '.'], ['Leifman', 'says', '200', 'years', 'ago', 'people', 'were', 'considered', '"', 'lunatics', '"', 'and', 'they', 'were', 'locked', 'up', 'in', 'jails', 'even', 'if', 'they', 'had', 'no', 'charges', 'against', 'them', '.'], ['They', 'were', 'just', 'considered', 'unfit', 'to', 'be', 'in', 'society', '.'], ['Over', 'the', 'years', ',', 'he', 'says', ',', 'there', 'was', 'some', 'public', 'outcry', ',', 'and', 'the', 'mentally', 'ill', 'were', 'moved', 'out', 'of', 'jails', 'and', 'into', 'hospitals', '.'], ['But', 'Leifman', 'says', 'many', 'of', 'these', 'mental', 'hospitals', 'were', 'so', 'horrible', 'they', 'were', 'shut', 'down', '.'], ['Where', 'did', 'the', 'patients', 'go', '?'], ['They', 'became', ',', 'in', 'many', 'cases', ',', 'the', 'homeless', ',', 'he', 'says', '.'], ['Leifman', 'says', 'in', '1955', 'there', 'were', 'more', 'than', 'half', 'a', 'million', 'people', 'in', 'state', 'mental', 'hospitals', ',', 'and', 'today', 'that', 'number', 'has', 'been', 'reduced', '90', 'percent', ',', 'and', '40,000', 'to', '50,000', 'people', 'are', 'in', 'mental', 'hospitals', '.'], ['The', 'judge', 'says', 'he', "'s", 'working', 'to', 'change', 'this', '.'], ['Starting', 'in', '2008', ',', 'many', 'inmates', 'who', 'would', 'otherwise', 'have', 'been', 'brought', 'to', 'the', '"', 'forgotten', 'floor', '"', ' ', 'will', 'instead', 'be', 'sent', 'to', 'a', 'new', 'mental', 'health', 'facility', '--', 'the', 'first', 'step', 'on', 'a', 'journey', 'toward', 'long', '-', 'term', 'treatment', ',', 'not', 'just', 'punishment', '.'], ['Leifman', 'says', 'it', "'s", 'not', 'the', 'complete', 'answer', ',', 'but', 'it', "'s", 'a', 'start', '.'], ['Leifman', 'says', 'the', 'best', 'part', 'is', 'that', 'it', "'s", 'a', 'win', '-', 'win', 'solution', '.'], ['The', 'patients', 'win', ',', 'the', 'families', 'are', 'relieved', ',', 'and', 'the', 'state', 'saves', 'money', 'by', 'simply', 'not', 'cycling', 'these', 'prisoners', 'through', 'again', 'and', 'again', '.'], ['And', ',', 'for', 'Leifman', ',', 'justice', 'is', 'served', '.'], ['E', '-', 'mail', 'to', 'a', 'friend', '.']], 'labels': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]}

First document in **CNN/DM** extractive **testing** set:

.. code-block::

    {'src': [['Marseille', ',', 'France', '(', 'CNN)The', 'French', 'prosecutor', 'leading', 'an', 'investigation', 'into', 'the', 'crash', 'of', 'Germanwings', 'Flight', '9525', 'insisted', 'Wednesday', 'that', 'he', 'was', 'not', 'aware', 'of', 'any', 'video', 'footage', 'from', 'on', 'board', 'the', 'plane', '.'], ['Marseille', 'prosecutor', 'Brice', 'Robin', 'told', 'CNN', 'that', '"', 'so', 'far', 'no', 'videos', 'were', 'used', 'in', 'the', 'crash', 'investigation', '.', '"'], ['He', 'added', ',', '"', 'A', 'person', 'who', 'has', 'such', 'a', 'video', 'needs', 'to', 'immediately', 'give', 'it', 'to', 'the', 'investigators', '.', '"'], ['Robin', "'s", 'comments', 'follow', 'claims', 'by', 'two', 'magazines', ',', 'German', 'daily', 'Bild', 'and', 'French', 'Paris', 'Match', ',', 'of', 'a', 'cell', 'phone', 'video', 'showing', 'the', 'harrowing', 'final', 'seconds', 'from', 'on', 'board', 'Germanwings', 'Flight', '9525', 'as', 'it', 'crashed', 'into', 'the', 'French', 'Alps', '.'], ['All', '150', 'on', 'board', 'were', 'killed', '.'], ['Paris', 'Match', 'and', 'Bild', 'reported', 'that', 'the', 'video', 'was', 'recovered', 'from', 'a', 'phone', 'at', 'the', 'wreckage', 'site', '.'], ['The', 'two', 'publications', 'described', 'the', 'supposed', 'video', ',', 'but', 'did', 'not', 'post', 'it', 'on', 'their', 'websites', '.'], ['The', 'publications', 'said', 'that', 'they', 'watched', 'the', 'video', ',', 'which', 'was', 'found', 'by', 'a', 'source', 'close', 'to', 'the', 'investigation', '.'], ['"', 'One', 'can', 'hear', 'cries', 'of', "'", 'My', 'God', "'", 'in', 'several', 'languages', ',', '"', 'Paris', 'Match', 'reported', '.'], ['"', 'Metallic', 'banging', 'can', 'also', 'be', 'heard', 'more', 'than', 'three', 'times', ',', 'perhaps', 'of', 'the', 'pilot', 'trying', 'to', 'open', 'the', 'cockpit', 'door', 'with', 'a', 'heavy', 'object', '.', ' '], ['Towards', 'the', 'end', ',', 'after', 'a', 'heavy', 'shake', ',', 'stronger', 'than', 'the', 'others', ',', 'the', 'screaming', 'intensifies', '.'], ['"', 'It', 'is', 'a', 'very', 'disturbing', 'scene', ',', '"', 'said', 'Julian', 'Reichelt', ',', 'editor', '-', 'in', '-', 'chief', 'of', 'Bild', 'online', '.'], ['An', 'official', 'with', 'France', "'s", 'accident', 'investigation', 'agency', ',', 'the', 'BEA', ',', 'said', 'the', 'agency', 'is', 'not', 'aware', 'of', 'any', 'such', 'video', '.'], ['Jean', '-', 'Marc', 'Menichini', ',', 'a', 'French', 'Gendarmerie', 'spokesman', 'in', 'charge', 'of', 'communications', 'on', 'rescue', 'efforts', 'around', 'the', 'Germanwings', 'crash', 'site', ',', 'told', 'CNN', 'that', 'the', 'reports', 'were', '"', 'completely', 'wrong', '"', 'and', '"', 'unwarranted', '.', '"'], ['Cell', 'phones', 'have', 'been', 'collected', 'at', 'the', 'site', ',', 'he', 'said', ',', 'but', 'that', 'they', '"', 'had', "n't", 'been', 'exploited', 'yet', '.'], ['Menichini', 'said', 'he', 'believed', 'the', 'cell', 'phones', 'would', 'need', 'to', 'be', 'sent', 'to', 'the', 'Criminal', 'Research', 'Institute', 'in', 'Rosny', 'sous', '-', 'Bois', ',', 'near', 'Paris', ',', 'in', 'order', 'to', 'be', 'analyzed', 'by', 'specialized', 'technicians', 'working', 'hand', '-', 'in', '-', 'hand', 'with', 'investigators', '.'], ['But', 'none', 'of', 'the', 'cell', 'phones', 'found', 'so', 'far', 'have', 'been', 'sent', 'to', 'the', 'institute', ',', 'Menichini', 'said', '.'], ['Asked', 'whether', 'staff', 'involved', 'in', 'the', 'search', 'could', 'have', 'leaked', 'a', 'memory', 'card', 'to', 'the', 'media', ',', 'Menichini', 'answered', 'with', 'a', 'categorical', '"', 'no', '.', '"'], ['Reichelt', 'told', '"', 'Erin', 'Burnett', ':'], ['Outfront', '"', 'that', 'he', 'had', 'watched', 'the', 'video', 'and', 'stood', 'by', 'the', 'report', ',', 'saying', 'Bild', 'and', 'Paris', 'Match', 'are', '"', 'very', 'confident', '"', 'that', 'the', 'clip', 'is', 'real', '.'], ['He', 'noted', 'that', 'investigators', 'only', 'revealed', 'they', "'d", 'recovered', 'cell', 'phones', 'from', 'the', 'crash', 'site', 'after', 'Bild', 'and', 'Paris', 'Match', 'published', 'their', 'reports', '.'], ['"', 'That', 'is', 'something', 'we', 'did', 'not', 'know', 'before', '.', '...'], ['Overall', 'we', 'can', 'say', 'many', 'things', 'of', 'the', 'investigation', 'were', "n't", 'revealed', 'by', 'the', 'investigation', 'at', 'the', 'beginning', ',', '"', 'he', 'said', '.'], ['German', 'airline', 'Lufthansa', 'confirmed', 'Tuesday', 'that', 'co', '-', 'pilot', 'Andreas', 'Lubitz', 'had', 'battled', 'depression', 'years', 'before', 'he', 'took', 'the', 'controls', 'of', 'Germanwings', 'Flight', '9525', ',', 'which', 'he', "'s", 'accused', 'of', 'deliberately', 'crashing', 'last', 'week', 'in', 'the', 'French', 'Alps', '.'], ['Lubitz', 'told', 'his', 'Lufthansa', 'flight', 'training', 'school', 'in', '2009', 'that', 'he', 'had', 'a', '"', 'previous', 'episode', 'of', 'severe', 'depression', ',', '"', 'the', 'airline', 'said', 'Tuesday', '.'], ['Email', 'correspondence', 'between', 'Lubitz', 'and', 'the', 'school', 'discovered', 'in', 'an', 'internal', 'investigation', ',', 'Lufthansa', 'said', ',', 'included', 'medical', 'documents', 'he', 'submitted', 'in', 'connection', 'with', 'resuming', 'his', 'flight', 'training', '.'], ['The', 'announcement', 'indicates', 'that', 'Lufthansa', ',', 'the', 'parent', 'company', 'of', 'Germanwings', ',', 'knew', 'of', 'Lubitz', "'s", 'battle', 'with', 'depression', ',', 'allowed', 'him', 'to', 'continue', 'training', 'and', 'ultimately', 'put', 'him', 'in', 'the', 'cockpit', '.'], ['Lufthansa', ',', 'whose', 'CEO', 'Carsten', 'Spohr', 'previously', 'said', 'Lubitz', 'was', '100', '%', 'fit', 'to', 'fly', ',', 'described', 'its', 'statement', 'Tuesday', 'as', 'a', '"', 'swift', 'and', 'seamless', 'clarification', '"', 'and', 'said', 'it', 'was', 'sharing', 'the', 'information', 'and', 'documents', '--', 'including', 'training', 'and', 'medical', 'records', '--', 'with', 'public', 'prosecutors', '.'], ['Spohr', 'traveled', 'to', 'the', 'crash', 'site', 'Wednesday', ',', 'where', 'recovery', 'teams', 'have', 'been', 'working', 'for', 'the', 'past', 'week', 'to', 'recover', 'human', 'remains', 'and', 'plane', 'debris', 'scattered', 'across', 'a', 'steep', 'mountainside', '.'], ['He', 'saw', 'the', 'crisis', 'center', 'set', 'up', 'in', 'Seyne', '-', 'les', '-', 'Alpes', ',', 'laid', 'a', 'wreath', 'in', 'the', 'village', 'of', 'Le', 'Vernet', ',', 'closer', 'to', 'the', 'crash', 'site', ',', 'where', 'grieving', 'families', 'have', 'left', 'flowers', 'at', 'a', 'simple', 'stone', 'memorial', '.'], ['Menichini', 'told', 'CNN', 'late', 'Tuesday', 'that', 'no', 'visible', 'human', 'remains', 'were', 'left', 'at', 'the', 'site', 'but', 'recovery', 'teams', 'would', 'keep', 'searching', '.'], ['French', 'President', 'Francois', 'Hollande', ',', 'speaking', 'Tuesday', ',', 'said', 'that', 'it', 'should', 'be', 'possible', 'to', 'identify', 'all', 'the', 'victims', 'using', 'DNA', 'analysis', 'by', 'the', 'end', 'of', 'the', 'week', ',', 'sooner', 'than', 'authorities', 'had', 'previously', 'suggested', '.'], ['In', 'the', 'meantime', ',', 'the', 'recovery', 'of', 'the', 'victims', "'", 'personal', 'belongings', 'will', 'start', 'Wednesday', ',', 'Menichini', 'said', '.'], ['Among', 'those', 'personal', 'belongings', 'could', 'be', 'more', 'cell', 'phones', 'belonging', 'to', 'the', '144', 'passengers', 'and', 'six', 'crew', 'on', 'board', '.'], ['Check', 'out', 'the', 'latest', 'from', 'our', 'correspondents', '.'], ['The', 'details', 'about', 'Lubitz', "'s", 'correspondence', 'with', 'the', 'flight', 'school', 'during', 'his', 'training', 'were', 'among', 'several', 'developments', 'as', 'investigators', 'continued', 'to', 'delve', 'into', 'what', 'caused', 'the', 'crash', 'and', 'Lubitz', "'s", 'possible', 'motive', 'for', 'downing', 'the', 'jet', '.'], ['A', 'Lufthansa', 'spokesperson', 'told', 'CNN', 'on', 'Tuesday', 'that', 'Lubitz', 'had', 'a', 'valid', 'medical', 'certificate', ',', 'had', 'passed', 'all', 'his', 'examinations', 'and', '"', 'held', 'all', 'the', 'licenses', 'required', '.', '"'], ['Earlier', ',', 'a', 'spokesman', 'for', 'the', 'prosecutor', "'s", 'office', 'in', 'Dusseldorf', ',', 'Christoph', 'Kumpa', ',', 'said', 'medical', 'records', 'reveal'], ['Lubitz', 'suffered', 'from', 'suicidal', 'tendencies', 'at', 'some', 'point', 'before', 'his', 'aviation', 'career', 'and', 'underwent', 'psychotherapy', 'before', 'he', 'got', 'his', 'pilot', "'s", 'license', '.'], ['Kumpa', 'emphasized', 'there', "'s", 'no', 'evidence', 'suggesting', 'Lubitz', 'was', 'suicidal', 'or', 'acting', 'aggressively', 'before', 'the', 'crash', '.'], ['Investigators', 'are', 'looking', 'into', 'whether', 'Lubitz', 'feared', 'his', 'medical', 'condition', 'would', 'cause', 'him', 'to', 'lose', 'his', 'pilot', "'s", 'license', ',', 'a', 'European', 'government', 'official', 'briefed', 'on', 'the', 'investigation', 'told', 'CNN', 'on', 'Tuesday', '.'], ['While', 'flying', 'was', '"', 'a', 'big', 'part', 'of', 'his', 'life'], [',', '"', 'the', 'source', 'said', ',', 'it', "'s", 'only', 'one', 'theory', 'being', 'considered', '.'], ['Another', 'source', ',', 'a', 'law', 'enforcement', 'official', 'briefed', 'on', 'the', 'investigation', ',', 'also', 'told', 'CNN', 'that', 'authorities', 'believe', 'the', 'primary', 'motive', 'for', 'Lubitz', 'to', 'bring', 'down', 'the', 'plane', 'was', 'that', 'he', 'feared', 'he', 'would', 'not', 'be', 'allowed', 'to', 'fly', 'because', 'of', 'his', 'medical', 'problems', '.'], ['Lubitz', "'s", 'girlfriend', 'told', 'investigators', 'he', 'had', 'seen', 'an', 'eye', 'doctor', 'and', 'a', 'neuropsychologist', ',', 'both', 'of', 'whom', 'deemed', 'him', 'unfit', 'to', 'work', 'recently', 'and', 'concluded', 'he', 'had', 'psychological', 'issues', ',', 'the', 'European', 'government', 'official', 'said', '.'], ['But', 'no', 'matter', 'what', 'details', 'emerge', 'about', 'his', 'previous', 'mental', 'health', 'struggles', ',', 'there', "'s", 'more', 'to', 'the', 'story', ',', 'said', 'Brian', 'Russell', ',', 'a', 'forensic', 'psychologist', '.'], ['"', 'Psychology', 'can', 'explain', 'why', 'somebody', 'would', 'turn', 'rage', 'inward', 'on', 'themselves', 'about', 'the', 'fact', 'that', 'maybe', 'they', 'were', "n't", 'going', 'to', 'keep', 'doing', 'their', 'job', 'and', 'they', "'re", 'upset', 'about', 'that'], ['and', 'so', 'they', "'re", 'suicidal', ',', '"', 'he', 'said', '.'], ['"', 'But', 'there', 'is', 'no', 'mental', 'illness', 'that', 'explains', 'why', 'somebody', 'then', 'feels', 'entitled', 'to', 'also', 'take', 'that', 'rage', 'and', 'turn', 'it', 'outward', 'on', '149', 'other', 'people', 'who', 'had', 'nothing', 'to', 'do', 'with', 'the', 'person', "'s", 'problems', '.', '"'], ['Who', 'was', 'the', 'captain', 'of', 'Germanwings', 'Flight', '9525', '?'], ['CNN', "'s", 'Margot', 'Haddad', 'reported', 'from', 'Marseille', 'and', 'Pamela', 'Brown', 'from', 'Dusseldorf', ',', 'while', 'Laura', 'Smith', '-', 'Spark', 'wrote', 'from', 'London', '.'], ['CNN', "'s", 'Frederik', 'Pleitgen', ',', 'Pamela', 'Boykoff', ',', 'Antonia', 'Mortensen', ',', 'Sandrine', 'Amiel', 'and', 'Anna', '-', 'Maja', 'Rappard', 'contributed', 'to', 'this', 'report', '.']], 'labels': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'tgt': 'Marseille prosecutor says " so far no videos were used in the crash investigation " despite media reports .<q>Journalists at Bild and Paris Match are " very confident " the video clip is real , an editor says .<q>Andreas Lubitz had informed his Lufthansa training school of an episode of severe depression , airline says .'}


Script Help
-----------

.. code-block::

    usage: convert_to_extractive.py [-h] [--base_output_path BASE_OUTPUT_PATH]
                                    [--split_names {train,val,test} [{train,val,test} ...]]
                                    [--add_target_to {train,val,test} [{train,val,test} ...]]
                                    [--source_ext SOURCE_EXT] [--target_ext TARGET_EXT]
                                    [--oracle_mode {greedy,combination}]
                                    [--shard_interval SHARD_INTERVAL]
                                    [--n_process N_PROCESS] [--batch_size BATCH_SIZE]
                                    [--compression] [--resume]
                                    [--tokenizer_log_interval TOKENIZER_LOG_INTERVAL]
                                    [--sentencizer] [--no_preprocess]
                                    [--min_sentence_ntokens MIN_SENTENCE_NTOKENS]
                                    [--max_sentence_ntokens MAX_SENTENCE_NTOKENS]
                                    [--min_example_nsents MIN_EXAMPLE_NSENTS]
                                    [--max_example_nsents MAX_EXAMPLE_NSENTS]
                                    [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                                    DIR

    Convert an Abstractive Summarization Dataset to the Extractive Task

    positional arguments:
    DIR                   path to data directory

    optional arguments:
    -h, --help            show this help message and exit
    --base_output_path BASE_OUTPUT_PATH
                            path to output processed data (default is `base_path`)
    --split_names {train,val,test} [{train,val,test} ...]
                            which splits of dataset to process
    --add_target_to {train,val,test} [{train,val,test} ...]
                            add the abstractive target to these splits (useful for
                            calculating rouge scores)
    --source_ext SOURCE_EXT
                            extension of source files
    --target_ext TARGET_EXT
                            extension of target files
    --oracle_mode {greedy,combination}
                            method to convert abstractive summaries to extractive
                            summaries
    --shard_interval SHARD_INTERVAL
                            how many examples to include in each shard of the dataset
                            (default: no shards)
    --n_process N_PROCESS
                            number of processes for multithreading
    --batch_size BATCH_SIZE
                            number of batches for tokenization
    --compression         use gzip compression when saving data
    --resume              resume from last shard
    --tokenizer_log_interval TOKENIZER_LOG_INTERVAL
                            minimum progress display update interval [default: 0.1]
                            seconds
    --sentencizer         use a spacy sentencizer instead of a statistical model for
                            sentence detection (much faster but less accurate); see
                            https://spacy.io/api/sentencizer
    --no_preprocess       do not run the preprocess function, which removes sentences
                            that are too long/short and examples that have too few/many
                            sentences
    --min_sentence_ntokens MIN_SENTENCE_NTOKENS
                            minimum number of tokens per sentence
    --max_sentence_ntokens MAX_SENTENCE_NTOKENS
                            maximum number of tokens per sentence
    --min_example_nsents MIN_EXAMPLE_NSENTS
                            minimum number of sentences per example
    --max_example_nsents MAX_EXAMPLE_NSENTS
                            maximum number of sentences per example
    -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --log {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                            Set the logging level (default: 'Info').
