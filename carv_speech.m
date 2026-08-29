function carv_speech(metadata, speechdir)

%% ============================================================
% ABLATION STUDY: TRANSITION-WINDOW EMOTION CHANGE DETECTION
% NN-ONLY VERSION
%
% Updated:
%   - Emotion similarity cleaning removed completely
%   - Change is detected between ANY two different emotions
%   - Same emotion      -> no change, label = 0
%   - Different emotion -> change, label = 1
%
% FEATURE LOGIC:
% Both change and no-change pairs require:
%   previous utterance >= TRANS_SEC
%   current utterance >= TRANS_SEC
%
% This keeps the signal duration similar for both classes:
%   Change    = TRANS_SEC + TRANS_SEC
%   No-change = 2 * TRANS_SEC
%
% Reports:
%   Train Accuracy, Train Weighted F1, Train F1 no-change, Train F1 change
%   Test  Accuracy, Test  Weighted F1, Test  F1 no-change, Test  F1 change
%
% Variants per fold:
%   1) Complex NN only : real + imaginary FFT transition features
%   2) Real-only NN   : real FFT transition features only
%   3) Imag-only NN   : imaginary FFT transition features only
%
% Requirements in same folder/path:
%   metadata.csv
%   Allsentences/ audio folder
%% ============================================================

%% =========================
% GLOBAL FIXED SEED
%% =========================
rng(42);

%% =========================
% 1. LOAD LABELS
%% =========================
data = readtable(metadata);

emotions = string(data.Emotion);
N = height(data);

% === EXTRACT SPEAKER AND DIALOGUE ID FROM ORIGINAL FILE ===
uttFiles = string(data.OriginalFile);

speaker = strings(N,1);
dialogID = strings(N,1);

for i = 1:N
    fname = erase(uttFiles(i), ".wav");
    parts = split(fname, "_");
    lastPart = parts(end);

    if startsWith(lastPart, "F")
        speaker(i) = "F";
    elseif startsWith(lastPart, "M")
        speaker(i) = "M";
    else
        speaker(i) = "";
    end

    if numel(parts) > 1
        dialogID(i) = strjoin(parts(1:end-1), "_");
    else
        dialogID(i) = fname;
    end
end

%% ============================================================
% SPEAKER-AWARE CHANGE LABELS
%
% Compare each utterance only with the previous utterance from
% the SAME speaker within the SAME dialogue.
%
% Logic:
%   Same emotion      -> no change, label = 0
%   Different emotion -> change, label = 1
%% ============================================================

change_labels = -1 * ones(N,1);
previous_same_speaker = nan(N,1);

lastF = [];
lastM = [];
currentDialog = "";

for i = 1:N

    if dialogID(i) ~= currentDialog
        currentDialog = dialogID(i);
        lastF = [];
        lastM = [];
    end

    if ismissing(emotions(i)) || strlength(emotions(i)) == 0
        continue;
    end

    if speaker(i) == "F"

        if ~isempty(lastF)
            change_labels(i) = double(~strcmp(emotions(i), emotions(lastF)));
            previous_same_speaker(i) = lastF;
        end

        lastF = i;

    elseif speaker(i) == "M"

        if ~isempty(lastM)
            change_labels(i) = double(~strcmp(emotions(i), emotions(lastM)));
            previous_same_speaker(i) = lastM;
        end

        lastM = i;
    end
end

fprintf('Female utterances: %d\n', sum(speaker == "F"));
fprintf('Male utterances: %d\n', sum(speaker == "M"));
fprintf('Valid same-speaker labelled transitions: %d\n', sum(change_labels ~= -1));
fprintf('Change transitions: %d\n', sum(change_labels == 1));
fprintf('No-change transitions: %d\n', sum(change_labels == 0));

%% =========================
% 2. AUDIO SETTINGS + LENGTH CHECK
%% =========================
audio_folder = 'speechdir';
audioFiles = string(data.AudioFile);

% Transition-window feature settings
TRANS_SEC = 4;
DOWNSAMPLE_FACTOR = 20;

% Fixed feature length after FFT padding/truncation
L = 100;

audio_length = zeros(N,1);
audio_fs = zeros(N,1);

for k = 1:N
    audioPath = fullfile(audio_folder, char(audioFiles(k)));

    if ~isfile(audioPath)
        error('Missing audio file: %s', audioPath);
    end

    info = audioinfo(audioPath);
    audio_length(k) = info.TotalSamples;
    audio_fs(k) = info.SampleRate;
end

fprintf('\n');
fprintf('TRANS_SEC: %.2f seconds\n', TRANS_SEC);
fprintf('Feature logic:\n');
fprintf('  Change pair    : previous %.2f sec + current %.2f sec\n', TRANS_SEC, TRANS_SEC);
fprintf('  No-change pair : previous %.2f sec + current %.2f sec\n', TRANS_SEC, TRANS_SEC);
fprintf('Fixed feature length L: %d\n', L);

%% =========================
% 3. CREATE SAME-SPEAKER PAIRS
%% =========================
X_pair = [];        % complex feature = [real; imaginary], padded/truncated to L
X_real_pair = [];   % real-only feature, padded/truncated to L
X_imag_pair = [];   % imaginary-only feature, padded/truncated to L

Y_pair = [];

original_pair_index = [];
pair_prev_index = [];
pair_curr_index = [];

pair_speaker = strings(0,1);
pair_dialogID = strings(0,1);

MAX_PAIR_LENGTH = 10000000000;

numDiscarded = 0;
discardedChange = 0;
discardedNoChange = 0;

% Counters for short utterances
numShortUtteranceDiscarded = 0;
shortUtteranceChange = 0;
shortUtteranceNoChange = 0;

idx = 1;

for i = 1:N

    prevIdx = previous_same_speaker(i);

    if isnan(prevIdx) || change_labels(i) == -1
      continue;
    end

    prevIdx = round(double(prevIdx));

    if prevIdx < 1 || prevIdx > length(emotions)
      continue;
    end

    prevEmotion = emotions(prevIdx);
    currEmotion = emotions(i);

    validTransition = ...
      (prevEmotion=="ang" & currEmotion=="ang") | ...
      (prevEmotion=="ang" & currEmotion=="fru") | ...
      (prevEmotion=="exc" & currEmotion=="hap") | ...
      (prevEmotion=="fru" & currEmotion=="ang") | ...
      (prevEmotion=="hap" & currEmotion=="exc") | ...
      (prevEmotion=="sad" & currEmotion=="fru") | ...
      (prevEmotion=="fru" & currEmotion=="sad");

    if ~validTransition
       continue;
    end

    if isnan(prevIdx) || change_labels(i) == -1
        continue;
    end

    prevIdx = double(prevIdx);

    % Safety checks: same speaker and same dialogue only
    if speaker(i) ~= speaker(prevIdx)
        continue;
    end

    if dialogID(i) ~= dialogID(prevIdx)
        continue;
    end

    isChange = change_labels(i);

    prevDurSec = audio_length(prevIdx) / audio_fs(prevIdx);
    currDurSec = audio_length(i)       / audio_fs(i);

    % ============================================================
    % UPDATED LENGTH CHECK
    %
    % Change pair:
    %   previous utterance must be at least TRANS_SEC
    %   current utterance must be at least TRANS_SEC
    %
    % No-change pair:
    %   bigger/longer utterance must be at least 2 * TRANS_SEC
    %
    % This keeps duration consistent:
    %   Change    = TRANS_SEC + TRANS_SEC
    %   No-change = 2 * TRANS_SEC
    % ============================================================
    
    % Both classes now use TRANS_SEC from EACH utterance

    tooShort = (prevDurSec < TRANS_SEC) || (currDurSec < TRANS_SEC);

    if tooShort

        numShortUtteranceDiscarded = numShortUtteranceDiscarded + 1;

        if isChange == 1 
            shortUtteranceChange = shortUtteranceChange + 1;
        else
            shortUtteranceNoChange = shortUtteranceNoChange + 1;
        end

        continue;
    end

    total_len = audio_length(i) + audio_length(prevIdx);

    if total_len > MAX_PAIR_LENGTH

        numDiscarded = numDiscarded + 1;

        if isChange == 1 
            discardedChange = discardedChange + 1;
        else
            discardedNoChange = discardedNoChange + 1;
        end

        continue;
    end

    prevAudioPath = fullfile(audio_folder, char(audioFiles(prevIdx)));
    currAudioPath = fullfile(audio_folder, char(audioFiles(i)));

    % ============================================================
    % FEATURE EXTRACTION
    %
    % Change:
    %   previous TRANS_SEC + current TRANS_SEC
    %
    % No-change:
    %   2 * TRANS_SEC from bigger/longer audio
    % ============================================================
    [feat_real, feat_imag, feat_complex] = extract_transition_fft_feature_parts( ...
        prevAudioPath, currAudioPath, TRANS_SEC, L, DOWNSAMPLE_FACTOR);

    X_real_pair(:,idx) = feat_real;
    X_imag_pair(:,idx) = feat_imag;
    X_pair(:,idx)      = feat_complex;

    Y_pair(idx) = change_labels(i);

    original_pair_index(idx) = i;
    pair_prev_index(idx) = prevIdx;
    pair_curr_index(idx) = i;

    pair_speaker(idx,1) = speaker(i);
    pair_dialogID(idx,1) = dialogID(i);

    idx = idx + 1;
end

Y_pair = Y_pair(:);
original_pair_index = original_pair_index(:);
pair_prev_index = pair_prev_index(:);
pair_curr_index = pair_curr_index(:);

%% ============================================================
% LYAPUNOV FEATURES ON FFT TRAJECTORY
%% ============================================================

numPairs = length(Y_pair);

V_energy   = zeros(numPairs,1);
R_energy   = ones(numPairs,1);
dV_energy  = zeros(numPairs,1);

eps_val = 1e-8;

for k = 2:numPairs

    sameSpeaker = pair_speaker(k) == pair_speaker(k-1);
    sameDialog  = pair_dialogID(k) == pair_dialogID(k-1);

    if sameSpeaker && sameDialog

        V_curr = sum(X_pair(:,k).^2);
        V_prev = sum(X_pair(:,k-1).^2);

        V_energy(k) = V_curr;

        R_energy(k) = V_curr / (V_prev + eps_val);

        dV_energy(k) = V_curr - V_prev;

    end
end

LyapFeatures = [ ...
    log1p(V_energy)'; ...
    log1p(abs(dV_energy))'; ...
    abs(R_energy-1)'; ...
    movmean(V_energy,30)' ...
];

X_pair_lyap = [X_pair; LyapFeatures];

fprintf('\nLyapunov features created\n');
fprintf('Base dimension      : %d\n', size(X_pair,1));
fprintf('Enhanced dimension  : %d\n', size(X_pair_lyap,1));

save('pair_information.mat', ...
    'pair_prev_index', ...
    'pair_curr_index', ...
    'Y_pair', ...
    'audioFiles', ...
    'emotions', ...
    'speaker', ...
    'dialogID');

fprintf('\n');
fprintf('Maximum pair length allowed: %d\n', MAX_PAIR_LENGTH);
fprintf('Discarded pairs due to maximum pair length: %d\n', numDiscarded);
fprintf('Discarded change pairs due to max length: %d\n', discardedChange);
fprintf('Discarded no-change pairs due to max length: %d\n', discardedNoChange);

fprintf('\n');
fprintf('Discarded pairs because pair did not satisfy required duration: %d\n', ...
    numShortUtteranceDiscarded);
fprintf('Short-duration discarded change pairs: %d\n', shortUtteranceChange);
fprintf('Short-duration discarded no-change pairs: %d\n', shortUtteranceNoChange);

fprintf('\nRemaining valid pairs: %d\n', length(Y_pair));

fprintf('\n=== FIRST SAME-SPEAKER PAIRS CHECK ===\n');

for kk = 1:min(20,length(Y_pair))

    pidx = pair_prev_index(kk);
    cidx = pair_curr_index(kk);

    if Y_pair(kk) == 1
        modeText = "CHANGE: previous TRANS_SEC + current TRANS_SEC";
    else
        modeText = "NO-CHANGE: previous TRANS_SEC + current TRANS_SEC";
    end

    fprintf('%s [%s] -> %s [%s], speaker=%s, label=%d, feature=%s\n', ...
        char(uttFiles(pidx)), char(emotions(pidx)), ...
        char(uttFiles(cidx)), char(emotions(cidx)), ...
        char(pair_speaker(kk)), Y_pair(kk), char(modeText));
end

%% =========================
% 4. MODEL PARAMETERS
%% =========================
alpha = 0.005;
epochs = 500;
lambda = 1e-4;

%% =========================
% 5. STRATIFIED K-FOLD
%% =========================
K = 5;

idx_0 = find(Y_pair == 0);
idx_1 = find(Y_pair == 1);

idx_0 = idx_0(randperm(length(idx_0)));
idx_1 = idx_1(randperm(length(idx_1)));

folds_0 = cell(K,1);
folds_1 = cell(K,1);

for k = 1:K
    folds_0{k} = idx_0(k:K:end);
    folds_1{k} = idx_1(k:K:end);
end

%% =========================
% 6. STORAGE FOR ABLATION
%% =========================

ablationNames = { ...
    'Complex NN only', ...
    'Complex NN + Lyapunov', ...
    'Real-only NN', ...
    'Imag-only NN'};

numAblations = numel(ablationNames);

test_acc_all = zeros(K,numAblations);
test_F1_all  = zeros(K,numAblations);
test_F10_all = zeros(K,numAblations);
test_F11_all = zeros(K,numAblations);

train_acc_all = zeros(K,numAblations);
train_F1_all  = zeros(K,numAblations);
train_F10_all = zeros(K,numAblations);
train_F11_all = zeros(K,numAblations);

fold_col = [];
model_col = {};

nTrain_col = [];
nTest_col  = [];

train_acc_col = [];
train_F1_col  = [];
train_F10_col = [];
train_F11_col = [];

test_acc_col = [];
test_F1_col  = [];
test_F10_col = [];
test_F11_col = [];


%% =========================
% 7. MAIN CROSS-VALIDATION LOOP
%% =========================
for fold = 1:K

    fprintf('\n========================================\n');
    fprintf('FOLD %d / %d\n', fold, K);
    fprintf('========================================\n');

    %% SPLIT
    test_idx = [folds_0{fold}; folds_1{fold}];

    train_idx = [];

    for k = 1:K
        if k ~= fold
            train_idx = [train_idx; folds_0{k}; folds_1{k}]; %#ok<AGROW>
        end
    end

    fprintf('Train samples: %d | Test samples: %d\n', length(train_idx), length(test_idx));
    fprintf('Train no-change/change: %d / %d\n', ...
        sum(Y_pair(train_idx)==0), sum(Y_pair(train_idx)==1));
    fprintf('Test  no-change/change: %d / %d\n', ...
        sum(Y_pair(test_idx)==0), sum(Y_pair(test_idx)==1));

 
%% ============================================================
% ABLATIONS
%% ============================================================

Xsets = cell(4,1);

Xsets{1} = X_pair;
Xsets{2} = X_pair_lyap;
Xsets{3} = X_real_pair;
Xsets{4} = X_imag_pair;

for ab = 1:4

    fprintf('\n--- Ablation %d: %s ---\n', ...
        ab, ablationNames{ab});

    X_current = Xsets{ab};

    X_train_all = X_current(:, train_idx);
    Y_train_all = Y_pair(train_idx);

    X_test = X_current(:, test_idx);
    Y_test = Y_pair(test_idx);

    %% NORMALISE

    muX = mean(X_train_all,2);
    sigmaX = std(X_train_all,0,2);
    sigmaX(sigmaX < 1e-6) = 1;

    X_train_all = (X_train_all - muX)./sigmaX;
    X_test = (X_test - muX)./sigmaX;

    %% BALANCE TRAINING SET

    % id0 = find(Y_train_all == 0);
    % id1 = find(Y_train_all == 1);
    % 
    % nMin = min(length(id0), length(id1));
    % 
    % id0 = id0(randperm(length(id0), nMin));
    % id1 = id1(randperm(length(id1), nMin));
    % 
    % train_bal = [id0; id1];
    % train_bal = train_bal(randperm(length(train_bal)));
    % 
    % X_train = X_train_all(:,train_bal);
    % Y_train = Y_train_all(train_bal);

    X_train = X_train_all;
    Y_train = Y_train_all;
    input_size_current = size(X_current,1) + 3;

    [w1_ab,w2_ab] = train_complex_nn_binary( ...
        X_train,...
        Y_train,...
        input_size_current,...
        alpha,...
        epochs,...
        lambda);

    %% TRAIN

    nn_prob_train = complex_nn_predict_scores( ...
        X_train_all,...
        w1_ab,...
        w2_ab);


    if ab == 2
        pred_train = nn_prob_train > 0.50;
    else
        pred_train = nn_prob_train > 0.50;
    end

    [acc_train,F1_train,F1_0_train,F1_1_train] = ...
        binary_metrics(Y_train_all,pred_train);

    %% TEST

    nn_prob_test = complex_nn_predict_scores( ...
        X_test,...
        w1_ab,...
        w2_ab);

    if ab == 2
        pred_test = nn_prob_test > 0.45;
    else
        pred_test = nn_prob_test > 0.50;
    end

    [acc_test,F1_test,F1_0_test,F1_1_test] = ...
        binary_metrics(Y_test,pred_test);


        % Store for final summaries
        train_acc_all(fold, ab) = acc_train;
        train_F1_all(fold, ab)  = F1_train;
        train_F10_all(fold, ab) = F1_0_train;
        train_F11_all(fold, ab) = F1_1_train;

        test_acc_all(fold, ab) = acc_test;
        test_F1_all(fold, ab)  = F1_test;
        test_F10_all(fold, ab) = F1_0_test;
        test_F11_all(fold, ab) = F1_1_test;

        fprintf('%s TRAIN Accuracy: %.2f%% | Train Weighted F1: %.3f | Train F1 no-change: %.3f | Train F1 change: %.3f\n', ...
            ablationNames{ab}, acc_train*100, F1_train, F1_0_train, F1_1_train);

        fprintf('%s TEST  Accuracy: %.2f%% | Test  Weighted F1: %.3f | Test  F1 no-change: %.3f | Test  F1 change: %.3f\n', ...
            ablationNames{ab}, acc_test*100, F1_test, F1_0_test, F1_1_test);

        % Store per-fold table rows
        fold_col(end+1,1) = fold; %#ok<SAGROW>
        model_col{end+1,1} = ablationNames{ab}; %#ok<SAGROW>

        nTrain_col(end+1,1) = length(Y_train_all); %#ok<SAGROW>
        nTest_col(end+1,1)  = length(Y_test); %#ok<SAGROW>

        train_acc_col(end+1,1) = acc_train; %#ok<SAGROW>
        train_F1_col(end+1,1)  = F1_train; %#ok<SAGROW>
        train_F10_col(end+1,1) = F1_0_train; %#ok<SAGROW>
        train_F11_col(end+1,1) = F1_1_train; %#ok<SAGROW>

        test_acc_col(end+1,1) = acc_test; %#ok<SAGROW>
        test_F1_col(end+1,1)  = F1_test; %#ok<SAGROW>
        test_F10_col(end+1,1) = F1_0_test; %#ok<SAGROW>
        test_F11_col(end+1,1) = F1_1_test; %#ok<SAGROW>

    end
end

%% =========================
% 8. FINAL ABLATION RESULTS PER FOLD
%% =========================
fprintf('\n============================\n');
fprintf('FINAL NN-ONLY ABLATION RESULTS PER FOLD\n');
fprintf('============================\n');

AblationResults = table( ...
    fold_col, ...
    model_col, ...
    nTrain_col, ...
    nTest_col, ...
    train_acc_col * 100, ...
    train_F1_col, ...
    train_F10_col, ...
    train_F11_col, ...
    test_acc_col * 100, ...
    test_F1_col, ...
    test_F10_col, ...
    test_F11_col, ...
    'VariableNames', {'Fold','Model','NTrain','NTest', ...
                      'TrainAccuracyPercent','TrainWeightedF1','TrainF1_NoChange','TrainF1_Change', ...
                      'TestAccuracyPercent','TestWeightedF1','TestF1_NoChange','TestF1_Change'});

disp(AblationResults);

writetable(AblationResults, 'ablation_results_per_fold.csv');

%% =========================
% 9. FINAL ABLATION SUMMARY
%% =========================
fprintf('\n============================\n');
fprintf('FINAL NN-ONLY ABLATION SUMMARY\n');
fprintf('============================\n');

summaryModel = {};

summaryTrainAccMean = [];
summaryTrainAccStd  = [];
summaryTrainF1Mean  = [];
summaryTrainF1Std   = [];
summaryTrainF10Mean = [];
summaryTrainF10Std  = [];
summaryTrainF11Mean = [];
summaryTrainF11Std  = [];

summaryTestAccMean = [];
summaryTestAccStd  = [];
summaryTestF1Mean  = [];
summaryTestF1Std   = [];
summaryTestF10Mean = [];
summaryTestF10Std  = [];
summaryTestF11Mean = [];
summaryTestF11Std  = [];

for ab = 1:numAblations

    summaryModel{end+1,1} = ablationNames{ab}; %#ok<SAGROW>

    summaryTrainAccMean(end+1,1) = mean(train_acc_all(:,ab)) * 100; %#ok<SAGROW>
    summaryTrainAccStd(end+1,1)  = std(train_acc_all(:,ab)) * 100; %#ok<SAGROW>

    summaryTrainF1Mean(end+1,1) = mean(train_F1_all(:,ab)); %#ok<SAGROW>
    summaryTrainF1Std(end+1,1)  = std(train_F1_all(:,ab)); %#ok<SAGROW>

    summaryTrainF10Mean(end+1,1) = mean(train_F10_all(:,ab)); %#ok<SAGROW>
    summaryTrainF10Std(end+1,1)  = std(train_F10_all(:,ab)); %#ok<SAGROW>

    summaryTrainF11Mean(end+1,1) = mean(train_F11_all(:,ab)); %#ok<SAGROW>
    summaryTrainF11Std(end+1,1)  = std(train_F11_all(:,ab)); %#ok<SAGROW>

    summaryTestAccMean(end+1,1) = mean(test_acc_all(:,ab)) * 100; %#ok<SAGROW>
    summaryTestAccStd(end+1,1)  = std(test_acc_all(:,ab)) * 100; %#ok<SAGROW>

    summaryTestF1Mean(end+1,1) = mean(test_F1_all(:,ab)); %#ok<SAGROW>
    summaryTestF1Std(end+1,1)  = std(test_F1_all(:,ab)); %#ok<SAGROW>

    summaryTestF10Mean(end+1,1) = mean(test_F10_all(:,ab)); %#ok<SAGROW>
    summaryTestF10Std(end+1,1)  = std(test_F10_all(:,ab)); %#ok<SAGROW>

    summaryTestF11Mean(end+1,1) = mean(test_F11_all(:,ab)); %#ok<SAGROW>
    summaryTestF11Std(end+1,1)  = std(test_F11_all(:,ab)); %#ok<SAGROW>
end

AblationSummary = table( ...
    summaryModel, ...
    summaryTrainAccMean, summaryTrainAccStd, ...
    summaryTrainF1Mean, summaryTrainF1Std, ...
    summaryTrainF10Mean, summaryTrainF10Std, ...
    summaryTrainF11Mean, summaryTrainF11Std, ...
    summaryTestAccMean, summaryTestAccStd, ...
    summaryTestF1Mean, summaryTestF1Std, ...
    summaryTestF10Mean, summaryTestF10Std, ...
    summaryTestF11Mean, summaryTestF11Std, ...
    'VariableNames', {'Model', ...
                      'TrainAccuracyMean','TrainAccuracyStd', ...
                      'TrainWeightedF1Mean','TrainWeightedF1Std', ...
                      'TrainF1_NoChangeMean','TrainF1_NoChangeStd', ...
                      'TrainF1_ChangeMean','TrainF1_ChangeStd', ...
                      'TestAccuracyMean','TestAccuracyStd', ...
                      'TestWeightedF1Mean','TestWeightedF1Std', ...
                      'TestF1_NoChangeMean','TestF1_NoChangeStd', ...
                      'TestF1_ChangeMean','TestF1_ChangeStd'});

disp(AblationSummary);

writetable(AblationSummary, 'ablation_summary.csv');

fprintf('\nSaved output files:\n');
fprintf('  ablation_results_per_fold.csv\n');
fprintf('  ablation_summary.csv\n');

%% ============================================================
% LOCAL FUNCTIONS
%% ============================================================

function [feat_real, feat_imag, feat_complex] = extract_transition_fft_feature_parts( ...
    prevAudioPath, currAudioPath, TRANS_SEC, L, DOWNSAMPLE_FACTOR)

    [yPrev, fsPrev] = audioread(prevAudioPath);
    [yCurr, fsCurr] = audioread(currAudioPath);

    % Convert stereo to mono if required
    yPrev = mean(yPrev, 2);
    yCurr = mean(yCurr, 2);

    % Match sampling rate of current audio to previous audio
    if fsCurr ~= fsPrev
        yCurr = resample(yCurr, fsPrev, fsCurr);
    end

    fs = fsPrev;

    Nwin = max(1, round(TRANS_SEC * fs));

    % ============================================================
    % FEATURE LOGIC
    %
    % Change pair:
    %   use last TRANS_SEC of previous audio
    %   + first TRANS_SEC of current audio
    %
    % No-change pair:
    %   use 2 * TRANS_SEC from the bigger/longer audio
    %
    % This makes both classes have approximately equal duration.
    % ============================================================

    % ============================================================
% SAME FEATURE CONSTRUCTION FOR CHANGE AND NO-CHANGE
%
% previous utterance -> last TRANS_SEC
% current utterance  -> first TRANS_SEC
%
% label only determines whether the emotion changed.
% ============================================================

prevStart = max(1, length(yPrev) - Nwin + 1);
prevTail  = yPrev(prevStart:end);

currEnd   = min(Nwin, length(yCurr));
currHead  = yCurr(1:currEnd);

transitionSignal = [prevTail; currHead];

    % Downsample transition signal
    transitionSignal = resample(transitionSignal, 1, DOWNSAMPLE_FACTOR);

    % Normalise time-domain signal
    transitionSignal = transitionSignal / (max(abs(transitionSignal)) + 1e-6);

    % FFT
    F = fft(transitionSignal);

    real_part = real(F);
    imag_part = imag(F);

    % Normalise real and imaginary parts
    real_part = real_part / (max(abs(real_part)) + 1e-6);
    imag_part = imag_part / (max(abs(imag_part)) + 1e-6);

    % Sigmoid squashing
    real_part = 1 ./ (1 + exp(-real_part));
    imag_part = 1 ./ (1 + exp(-imag_part));

    % Complex feature = real followed by imaginary, padded/truncated to L
    feat_full = [real_part; imag_part];

    if length(feat_full) >= L
        feat_complex = feat_full(1:L);
    else
        feat_complex = [feat_full; zeros(L - length(feat_full), 1)];
    end

    % Real-only feature
    if length(real_part) >= L
        feat_real = real_part(1:L);
    else
        feat_real = [real_part; zeros(L - length(real_part), 1)];
    end

    % Imag-only feature
    if length(imag_part) >= L
        feat_imag = imag_part(1:L);
    else
        feat_imag = [imag_part; zeros(L - length(imag_part), 1)];
    end

    feat_real = feat_real(:);
    feat_imag = feat_imag(:);
    feat_complex = feat_complex(:);
end

function [w1, w2] = train_complex_nn_binary(X_train, Y_train, input_size, alpha, epochs, lambda)

    w1 = randn(input_size,1) * 0.01;
    w2 = randn(input_size,1) * 0.01;

    num_samples = length(Y_train);

    for epoch = 1:epochs

        idxp = randperm(num_samples);

        for ii = 1:num_samples

            i = idxp(ii);

            x = X_train(:,i);

            % Per-sample normalisation
            x = (x - mean(x)) / (std(x) + 1e-6);

            Uin = [0; 0; 1; x];

            z1 = w1' * Uin;
            z2 = w2' * Uin;

            z1 = max(min(z1,10),-10);
            z2 = max(min(z2,10),-10);

            a1 = 1/(1+exp(-z1));
            a2 = 1/(1+exp(-z2));

            yhat = (a1 + a2) / 2;

            e = Y_train(i) - yhat;

            w1 = w1 + alpha * (e * Uin - lambda * w1);
            w2 = w2 + alpha * (e * Uin - lambda * w2);
        end
    end
end

function prob = complex_nn_predict_scores(Xdata, w1, w2)

    n = size(Xdata,2);
    prob = zeros(n,1);

    for i = 1:n

        x = Xdata(:,i);

        % Per-sample normalisation
        x = (x - mean(x)) / (std(x) + 1e-6);

        Uin = [0; 0; 1; x];

        z1 = w1' * Uin;
        z2 = w2' * Uin;

        z1 = max(min(z1,10),-10);
        z2 = max(min(z2,10),-10);

        a1 = 1/(1+exp(-z1));
        a2 = 1/(1+exp(-z2));

        prob(i) = (a1 + a2) / 2;
    end
end

function [acc, F1_weighted, F1_0, F1_1] = binary_metrics(yTrue, yPred)

    yTrue = double(yTrue(:));
    yPred = double(yPred(:));

    TP = sum((yTrue == 1) & (yPred == 1));
    TN = sum((yTrue == 0) & (yPred == 0));
    FP = sum((yTrue == 0) & (yPred == 1));
    FN = sum((yTrue == 1) & (yPred == 0));

    acc = (TP + TN) / max(length(yTrue),1);

    % F1 for class 1: change
    prec1 = TP / (TP + FP + eps);
    rec1  = TP / (TP + FN + eps);
    F1_1  = 2*(prec1*rec1)/(prec1 + rec1 + eps);

    % F1 for class 0: no-change
    prec0 = TN / (TN + FN + eps);
    rec0  = TN / (TN + FP + eps);
    F1_0  = 2*(prec0*rec0)/(prec0 + rec0 + eps);

    n0 = sum(yTrue == 0);
    n1 = sum(yTrue == 1);

    F1_weighted = (n0*F1_0 + n1*F1_1) / max(n0+n1,1);
end