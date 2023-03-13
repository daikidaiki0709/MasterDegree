%% �萔���w��
clear
% ��͗̈�i�P�ʁF�s�N�Z���j
rad=400;

% �����ɂP�s�N�Z���ɑΉ�������ۂ̒������L��
% �ȉ��̗�ł͎��ۂ̒�����100mm�̂��̂̉�f����1637
pixel_length = 1637;
actual_length = 100;

% �P�s�N�Z���ɑΉ�������ۂ̒���
length_per_pixel=actual_length/pixel_length;

% �_�[�N�̈�̕��ϋP�x
ave_dark=[801.4,801.5,801.5,801.7,801.7,801.8,802.9,805.9];

% �I�����Ԃ̈Ⴂ�𒲐����邽�߂̌W��
exp_coeff=[1000, 10^2.5, 10^2, 10^1.5, 10,10^0.5,1,10^(-0.5)];

% �v���t�@�C���̃m�C�Y�����Ɏg�p����臒l�i�v���t�@�C�������܂��`��ł���ΕύX�̕K�v���Ȃ��j
CUT_MIN = 810;
CUT_MAX = 65441;

%% �������Ԃ�ݒ肷��
condition = "SecondStorage";
storage_period = "00";
%% ��̓X�N���v�g

tic
test = [];
sample_name = [];
for wavelength = ["633nm","850nm"]% �g���g����I�� 
    for sample_num = ["01","02","03","04","05","06","07","08","09","10"]% �T���v���̐��i��񂲂̐��j���i�[ 
        for point_num = ["1","2","3","4"]% �̓��̏Ǝ˓_�̐�

            %%%%%%%%%%%%%%%%%%%% loop����path�������i�����ŃG���[���o�Ȃ���Ί�{�I�Ɍ��̑���͂��܂������͂��A����΂��ĉ�ǂ��āI�I�j %%%%%%%%%%%%%%%%%%%%
            % �T���v�����i�摜�̃t�@�C�����j�̒�`
            % �T���v������"��������_�T���v���ԍ�_�Ǝ˓_�ԍ�"�Ƃ���
                % �������ԁF4�T�Ԓ����Ȃ�"04"�Ȃǁistorage_period�Œ�`���Ă���j
                % �T���v���ԍ��F�̔ԍ��i���[�U�[�v���O�Ɏ����Ŋ���U��j
                % �Ǝ˓_�ԍ��F�̓��ŕ������v�������ꍇ�̒ʂ��ԍ��i��񂲂Ȃ�1�̂ɂ�4�_�v�����Ă����̂ŁA1,2,3,4�ƒ�`�j
            sample_name_temp = append(storage_period,'_',sample_num,'_',point_num);
    
            % �摜���ۑ�����Ă���f�B���N�g�����w��
            % path_folder�̃t�H���_�\���F�f�[�^���i�[���Ă���t�H���_\�����O���[�v\�g��\��������
                % �f�[�^���i�[���Ă���t�H���_�FC:\Users\Mito Kokawa\Documents\MATLAB\IidaDaiki\2022_apple_study\laser_data\image_data\
                % �����O���[�v�FSecondStorage�icondition�ŏ�Œ�`���Ă���j
                % �g���F633nm or 850nm�iwavelength�Œ�`���Ă���j
                % �������ԁF4�T�Ԓ����Ȃ�"04"�Ȃǁistorage_period�Œ�`���Ă���j
            path_folder = append("C:\Users\Mito Kokawa\Documents\MATLAB\IidaDaiki\2022_apple_study\laser_data\image_data\",condition,"\",wavelength,"\week",storage_period);

            % path_temp�̃t�H���_�\���F�摜�f�[�^�ꗗ\�����O���[�v\�g��\��������\��������\�X�̃T���v��\8���̉摜
                % ��. C:\Users\Mito Kokawa\Documents\MATLAB\IidaDaiki\2022_apple_study\laser_data\image_data\SecondStorage\633nm\week00\00_01_4
                % �ŏI�I�Ɋe�T���v���̃��[�U�[�U�����f�[�^�i�I�����ԕ��̉摜�Q�t�H���_�ɂ��ǂ蒅����Ηǂ��I�j
            path_temp = append(path_folder,"\",sample_name_temp);
    
            % path_temp���J�����g�f�B���N�g���֕ύX
            cd(path_temp);
            

            %%%%%%%%%%%%%%%%%%%% �摜�̓ǂݍ��݂���уm�C�Y���� %%%%%%%%%%%%%%%%%%%%
            % �t�H���_���ɂ���C�ȉ��̌`���̃f�[�^��ǂݍ���
            dirOutput_tif = dir('*.TIFF');
            dirOutput_tiff = dir('*.TIF');
            dirOutput_png = dir('*.png');
            
            Filenames_tiff = {dirOutput_tiff.name}'; 
            Filenames_tif = {dirOutput_tif.name}';
            Filenames_png = {dirOutput_png.name}';
            Filenames=cat(1,Filenames_tif,Filenames_tiff,Filenames_png);
            imagenum=size(Filenames,1);
            
            % �摜�T�C�Y���擾
            I=imread(Filenames{1,1});
            [imagesizeY,imagesizeX]=size(I);
            imagedata=zeros(imagesizeY,imagesizeX,imagenum);
            
            % �t�H���_���̉摜�̓ǂݍ��݁E�m�C�Y����
            disp('�f�[�^��ǂݍ��ݒ����'); 
            for i=1:imagenum
                disp(Filenames{i,1})
                I=imread(Filenames{i,1}); 
                I_median = medfilt2(I,[5,5]); %5�~5��median filter�K�p�i�m�C�Y�����̖ړI�j
                imagedata(:,:,i)=double(I_median);
            end


            %%%%%%%%%%%%%%%%%%%% ���S�_�̌��o�i�V�������@�j%%%%%%%%%%%%%%%%%%%%

            % �܂���2�l��(��Ö@)
            temp = cast(imagedata(:,:,5),"uint16"); % �f�[�^�^�̕ύX
            temp_bina = imbinarize(temp); % 2�l��
            temp_bina = imfill(temp_bina,'holes'); % ���̖��ߍ���
            
            % �I�u�W�F�N�g���o
            stats = regionprops(temp_bina,'area','centroid'); % �I�u�W�F�N�g�̌��o���ʂ��i�[
            circ_ind=[stats.Area] >= max([stats.Area]); % ���o���ꂽ�I�u�W�F�N�g�̒��ōő�ʐρi���˓_�t��)�̂��̂��Z�o
            
            % ���˓_�̈ʒu�i���S�_�j���Z�o
            circ=stats(circ_ind,:);
            center = floor(circ.Centroid);


            %%%%%%%%%%%%%%%%%%%% ���S�_�Ƃ̊e��f�̋����E�P�x�i��f�l�j���Z�o�@%%%%%%%%%%%%%%%%%%%%

            % ��͗̈�(ROI)�Ɋ�Â����摜�̐��`
            analysis_image=squeeze(imagedata(center(2)-rad:center(2)+rad,...
                center(1)-rad:center(1)+rad,:));
            
            % ���S�_����̋����ɂ�����P�x�l���Z�o
            dist_int_matrix=zeros((rad*2+1)^2,imagenum+1);
            for i=1:size(analysis_image,1)
                for j=1:size(analysis_image,2)
                    dist_int_matrix((i-1)*(rad*2+1)+j,1)=((i-rad)^2+(j-rad)^2)^(1/2);
                    for k=1:imagenum
                        % MAX�̒l���傫���l��NaN�ɒu���i�m�C�Y�����ɗL���j
                        if analysis_image(i,j,k)>CUT_MAX
                            dist_int_matrix((i-1)*(rad*2+1)+j,k+1)=nan;
                        % MIN�̒l��菬�����l��NaN�ɒu���i�m�C�Y�����ɗL���j
                        elseif analysis_image(i,j,k)<CUT_MIN
                            dist_int_matrix((i-1)*(rad*2+1)+j,k+1)=nan;
                        else
                            dist_int_matrix((i-1)*(rad*2+1)+j,k+1)=analysis_image(i,j,k);  
                        end
                    end
                end
            end
            
            % ���S�_����̋����̏��ɕ��ׂ�i�\�[�g�j
            dist_int_matrix_sort=sortrows(dist_int_matrix,1);
            
            % ���������ɂ�����P�x�l�𕽋ω�
            % Table�f�[�^�ɕϊ��i���ϒl�̌v�Z�����₷�����߁j
            varnames={'Distance','Intensity'};
            dist_int_matrix_T=table(dist_int_matrix_sort(:,1),dist_int_matrix_sort(:,2:end),'VariableNames',varnames);
            func=@(x) mean(x,1);
            dist_meanint=varfun(func,dist_int_matrix_T,'InputVariables','Intensity',...
                'GroupingVariables','Distance');
            distance=dist_meanint.Distance;
            actual_distance = distance * length_per_pixel;
            

            %%%%%%%%%%%%%%%%%%%% �I�����Ԃł̋K�i������уv���t�@�C����HDR�����@%%%%%%%%%%%%%%%%%%%%
            % �����ł͂܂��I�����Ԃ��قȂ�P�x�v���t�@�C���͕ʂ̃f�[�^�Ƃ��ĕۑ�����Ă���
            % �v���t�@�C���쐬�̂��߂ɁA�e�I�����Ԃł̈Ód���m�C�Y���e�摜���獷�������i�Ód���͘I�����Ԃɔ�Ⴕ�Ȃ����߁j
            intprofile=dist_meanint.Fun_Intensity-repmat(ave_dark,[size(distance,1) 1]);
            intprofile(intprofile(:)<0)=nan;
            
            % �v���t�@�C���̍쐬 (�e�I�����Ԃ̋t���~�e�摜�̕��ρj
            % allintprofile_log: log�ϊ���
            allintprofile_log=mean(log10(intprofile .* repmat(exp_coeff,[size(distance,1) 1])),2,'omitnan');

            % ��͌��ʂ��i�[
            test = [test,allintprofile_log];
            sample_name = [sample_name sample_name_temp];

            %%%%%%%%%%%%%%%%%%%% HDR�摜�̕ۑ��iMATLAB�̊֐��j�@%%%%%%%%%%%%%%%%%%%%�j
            imagedata_hdr = zeros(imagesizeY,imagesizeX,imagenum);
            % �e�摜����Ή�����Ód�����������Auint16�ɐ��`
            for i=1:imagenum
                imagedata_hdr(:,:,i) = (imagedata(:,:,i) - ave_dark(:,i));
                imagedata_hdr(:,:,i) = uint16(imagedata_hdr(:,:,i));
            end

            % HDR�摜�̍쐬�i���ω��j
            % ��͂̂��߂�imagedata_hdr��cell�`���ɕύX
            imagedata_hdr_cell = {};
            for i=1:imagenum
                imagedata_hdr_cell{i} = imagedata_hdr(:,:,i);
                imagedata_hdr_cell{i} = uint16(imagedata_hdr(:,:,i));
            end
            % MATLAB��HDR�����ł���֐�������
                % �������I�����ԂȂ̂ŁAexp_coeff�̋t����n��
            image_HDR = makehdr(imagedata_hdr_cell,'RelativeExposure',1./exp_coeff);
            image_HDR = tonemap(image_HDR);

            % ��͂Ɏg�p����͈͂͂����悻�A���a������35mm�܂ł̂��߁A����p�ɉ��H����
            image_HDR = image_HDR(:,360:1560);
            
            % HDR�摜�̕ۑ�
                % ���͉摜�̃p�X�̂悤�Ɏ����Ő݌v����ΔC�ӂ̏ꏊ�ɉ摜��ۑ��\
            % path_hdr = append("C:\Users\Mito Kokawa\Documents\MATLAB\IidaDaiki\2022_apple_study\laser_data\HDRimage\MATLAB_func\",condition,"\",wavelength,"\week",storage_period,"\");
            % imwrite(image_HDR, append(path_hdr,sample_name_temp,".png"));

            clc

        end 
    end

    % csv���̂��߂̐��^
    test = [sample_name;test];
    actual_distance = [("distance (mm)"); actual_distance];
    test = [actual_distance test];
    
    % �v���t�@�C���̕ۑ�
        % ���͉摜�̃p�X�̂悤�Ɏ����Ő݌v����ΔC�ӂ̏ꏊ�Ƀv���t�@�C����ۑ��\
    % writematrix(test,append("C:\Users\Mito Kokawa\Documents\MATLAB\IidaDaiki\2022_apple_study\laser_data\Profile_new\Profile_",condition,"_",storage_period,"_",wavelength,".csv"))

    % �ϐ���������
    test = [];
    sample_name = [];

end

toc
disp("finish")