%% 定数を指定
clear
% 解析領域（単位：ピクセル）
rad=400;

% ここに１ピクセルに対応する実際の長さを記載
% 以下の例では実際の長さが100mmのものの画素数が1637
pixel_length = 1637;
actual_length = 100;

% １ピクセルに対応する実際の長さ
length_per_pixel=actual_length/pixel_length;

% ダーク領域の平均輝度
ave_dark=[801.4,801.5,801.5,801.7,801.7,801.8,802.9,805.9];

% 露光時間の違いを調整するための係数
exp_coeff=[1000, 10^2.5, 10^2, 10^1.5, 10,10^0.5,1,10^(-0.5)];

% プロファイルのノイズ除去に使用する閾値（プロファイルがうまく描画できれば変更の必要性なし）
CUT_MIN = 810;
CUT_MAX = 65441;

%% 貯蔵期間を設定する
condition = "SecondStorage";
storage_period = "00";
%% 解析スクリプト

tic
test = [];
sample_name = [];
for wavelength = ["633nm","850nm"]% 使う波長を選択 
    for sample_num = ["01","02","03","04","05","06","07","08","09","10"]% サンプルの数（りんごの数）を格納 
        for point_num = ["1","2","3","4"]% 個体内の照射点の数

            %%%%%%%%%%%%%%%%%%%% loop内のpathを完成（ここでエラーが出なければ基本的に後ろの操作はうまくいくはず、がんばって解読して！！） %%%%%%%%%%%%%%%%%%%%
            % サンプル名（画像のファイル名）の定義
            % サンプル名は"貯蔵期間_サンプル番号_照射点番号"とする
                % 貯蔵期間：4週間貯蔵なら"04"など（storage_periodで定義している）
                % サンプル番号：個体番号（レーザー計測前に自分で割り振る）
                % 照射点番号：個体内で複数個所計測した場合の通し番号（りんごなら1個体につき4点計測していたので、1,2,3,4と定義）
            sample_name_temp = append(storage_period,'_',sample_num,'_',point_num);
    
            % 画像が保存されているディレクトリを指定
            % path_folderのフォルダ構造：データを格納しているフォルダ\貯蔵グループ\波長\貯蔵期間
                % データを格納しているフォルダ：C:\Users\Mito Kokawa\Documents\MATLAB\IidaDaiki\2022_apple_study\laser_data\image_data\
                % 貯蔵グループ：SecondStorage（conditionで上で定義している）
                % 波長：633nm or 850nm（wavelengthで定義している）
                % 貯蔵期間：4週間貯蔵なら"04"など（storage_periodで定義している）
            path_folder = append("C:\Users\Mito Kokawa\Documents\MATLAB\IidaDaiki\2022_apple_study\laser_data\image_data\",condition,"\",wavelength,"\week",storage_period);

            % path_tempのフォルダ構造：画像データ一覧\貯蔵グループ\波長\貯蔵期間\貯蔵期間\個々のサンプル\8枚の画像
                % 例. C:\Users\Mito Kokawa\Documents\MATLAB\IidaDaiki\2022_apple_study\laser_data\image_data\SecondStorage\633nm\week00\00_01_4
                % 最終的に各サンプルのレーザー散乱生データ（露光時間分の画像群フォルダにたどり着ければ良し！）
            path_temp = append(path_folder,"\",sample_name_temp);
    
            % path_tempをカレントディレクトリへ変更
            cd(path_temp);
            

            %%%%%%%%%%%%%%%%%%%% 画像の読み込みおよびノイズ除去 %%%%%%%%%%%%%%%%%%%%
            % フォルダ内にある，以下の形式のデータを読み込む
            dirOutput_tif = dir('*.TIFF');
            dirOutput_tiff = dir('*.TIF');
            dirOutput_png = dir('*.png');
            
            Filenames_tiff = {dirOutput_tiff.name}'; 
            Filenames_tif = {dirOutput_tif.name}';
            Filenames_png = {dirOutput_png.name}';
            Filenames=cat(1,Filenames_tif,Filenames_tiff,Filenames_png);
            imagenum=size(Filenames,1);
            
            % 画像サイズを取得
            I=imread(Filenames{1,1});
            [imagesizeY,imagesizeX]=size(I);
            imagedata=zeros(imagesizeY,imagesizeX,imagenum);
            
            % フォルダ内の画像の読み込み・ノイズ除去
            disp('データを読み込み中･･･'); 
            for i=1:imagenum
                disp(Filenames{i,1})
                I=imread(Filenames{i,1}); 
                I_median = medfilt2(I,[5,5]); %5×5のmedian filter適用（ノイズ除去の目的）
                imagedata(:,:,i)=double(I_median);
            end


            %%%%%%%%%%%%%%%%%%%% 中心点の検出（新しい方法）%%%%%%%%%%%%%%%%%%%%

            % まずは2値化(大津法)
            temp = cast(imagedata(:,:,5),"uint16"); % データ型の変更
            temp_bina = imbinarize(temp); % 2値化
            temp_bina = imfill(temp_bina,'holes'); % 穴の埋め込み
            
            % オブジェクト検出
            stats = regionprops(temp_bina,'area','centroid'); % オブジェクトの検出結果を格納
            circ_ind=[stats.Area] >= max([stats.Area]); % 検出されたオブジェクトの中で最大面積（入射点付近)のものを算出
            
            % 入射点の位置（中心点）を算出
            circ=stats(circ_ind,:);
            center = floor(circ.Centroid);


            %%%%%%%%%%%%%%%%%%%% 中心点との各画素の距離・輝度（画素値）を算出　%%%%%%%%%%%%%%%%%%%%

            % 解析領域(ROI)に基づいた画像の整形
            analysis_image=squeeze(imagedata(center(2)-rad:center(2)+rad,...
                center(1)-rad:center(1)+rad,:));
            
            % 中心点からの距離における輝度値を算出
            dist_int_matrix=zeros((rad*2+1)^2,imagenum+1);
            for i=1:size(analysis_image,1)
                for j=1:size(analysis_image,2)
                    dist_int_matrix((i-1)*(rad*2+1)+j,1)=((i-rad)^2+(j-rad)^2)^(1/2);
                    for k=1:imagenum
                        % MAXの値より大きい値はNaNに置換（ノイズ除去に有効）
                        if analysis_image(i,j,k)>CUT_MAX
                            dist_int_matrix((i-1)*(rad*2+1)+j,k+1)=nan;
                        % MINの値より小さい値はNaNに置換（ノイズ除去に有効）
                        elseif analysis_image(i,j,k)<CUT_MIN
                            dist_int_matrix((i-1)*(rad*2+1)+j,k+1)=nan;
                        else
                            dist_int_matrix((i-1)*(rad*2+1)+j,k+1)=analysis_image(i,j,k);  
                        end
                    end
                end
            end
            
            % 中心点からの距離の順に並べる（ソート）
            dist_int_matrix_sort=sortrows(dist_int_matrix,1);
            
            % 同じ距離における輝度値を平均化
            % Tableデータに変換（平均値の計算がしやすくため）
            varnames={'Distance','Intensity'};
            dist_int_matrix_T=table(dist_int_matrix_sort(:,1),dist_int_matrix_sort(:,2:end),'VariableNames',varnames);
            func=@(x) mean(x,1);
            dist_meanint=varfun(func,dist_int_matrix_T,'InputVariables','Intensity',...
                'GroupingVariables','Distance');
            distance=dist_meanint.Distance;
            actual_distance = distance * length_per_pixel;
            

            %%%%%%%%%%%%%%%%%%%% 露光時間での規格化およびプロファイルのHDR合成　%%%%%%%%%%%%%%%%%%%%
            % ここではまだ露光時間が異なる輝度プロファイルは別のデータとして保存されている
            % プロファイル作成のために、各露光時間での暗電流ノイズを各画像から差し引く（暗電流は露光時間に比例しないため）
            intprofile=dist_meanint.Fun_Intensity-repmat(ave_dark,[size(distance,1) 1]);
            intprofile(intprofile(:)<0)=nan;
            
            % プロファイルの作成 (各露光時間の逆数×各画像の平均）
            % allintprofile_log: log変換後
            allintprofile_log=mean(log10(intprofile .* repmat(exp_coeff,[size(distance,1) 1])),2,'omitnan');

            % 解析結果を格納
            test = [test,allintprofile_log];
            sample_name = [sample_name sample_name_temp];

            %%%%%%%%%%%%%%%%%%%% HDR画像の保存（MATLABの関数）　%%%%%%%%%%%%%%%%%%%%）
            imagedata_hdr = zeros(imagesizeY,imagesizeX,imagenum);
            % 各画像から対応する暗電流を除去し、uint16に整形
            for i=1:imagenum
                imagedata_hdr(:,:,i) = (imagedata(:,:,i) - ave_dark(:,i));
                imagedata_hdr(:,:,i) = uint16(imagedata_hdr(:,:,i));
            end

            % HDR画像の作成（平均化）
            % 解析のためにimagedata_hdrをcell形式に変更
            imagedata_hdr_cell = {};
            for i=1:imagenum
                imagedata_hdr_cell{i} = imagedata_hdr(:,:,i);
                imagedata_hdr_cell{i} = uint16(imagedata_hdr(:,:,i));
            end
            % MATLABでHDR合成できる関数がある
                % 引数が露光時間なので、exp_coeffの逆数を渡す
            image_HDR = makehdr(imagedata_hdr_cell,'RelativeExposure',1./exp_coeff);
            image_HDR = tonemap(image_HDR);

            % 解析に使用する範囲はおおよそ、動径距離が35mmまでのため、それ用に加工する
            image_HDR = image_HDR(:,360:1560);
            
            % HDR画像の保存
                % 入力画像のパスのように自分で設計すれば任意の場所に画像を保存可能
            % path_hdr = append("C:\Users\Mito Kokawa\Documents\MATLAB\IidaDaiki\2022_apple_study\laser_data\HDRimage\MATLAB_func\",condition,"\",wavelength,"\week",storage_period,"\");
            % imwrite(image_HDR, append(path_hdr,sample_name_temp,".png"));

            clc

        end 
    end

    % csv化のための成型
    test = [sample_name;test];
    actual_distance = [("distance (mm)"); actual_distance];
    test = [actual_distance test];
    
    % プロファイルの保存
        % 入力画像のパスのように自分で設計すれば任意の場所にプロファイルを保存可能
    % writematrix(test,append("C:\Users\Mito Kokawa\Documents\MATLAB\IidaDaiki\2022_apple_study\laser_data\Profile_new\Profile_",condition,"_",storage_period,"_",wavelength,".csv"))

    % 変数を初期化
    test = [];
    sample_name = [];

end

toc
disp("finish")