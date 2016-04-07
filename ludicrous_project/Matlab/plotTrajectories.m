fileID = fopen('groundtruth.txt');
groundtruth = textscan(fileID,'%f %f %f %f %f %f %f %f','CommentStyle','#');
fclose(fileID);

fileID = fopen('rgbd_dataset_freiburg1_deskour_trajectory.txt');
our_result = textscan(fileID,'%f %f %f %f %f %f %f %f','CommentStyle','#');
fclose(fileID);

fileID = fopen('rgbd_dataset_freiburg1_desk_our_trajectory_WITH_weights.txt');
our_result_weighted = textscan(fileID,'%f %f %f %f %f %f %f %f','CommentStyle','#');
fclose(fileID);

gt_timestamps  =groundtruth{1,1}(356:end);
gt_tx          =groundtruth{1,2}(356:end);
gt_ty          =groundtruth{1,3}(356:end);
gt_tz          =groundtruth{1,4}(356:end);
gt_qx          =groundtruth{1,5}(356:end);
gt_qy          =groundtruth{1,6}(356:end);
gt_qz          =groundtruth{1,7}(356:end);
gt_qw          =groundtruth{1,8}(356:end);

res_timestamps  =our_result{1,1};
res_tx          =our_result{1,2};
res_ty          =our_result{1,3};
res_tz          =our_result{1,4};
res_qx          =our_result{1,5};
res_qy          =our_result{1,6};
res_qz          =our_result{1,7};
res_qw          =our_result{1,8};

rw_timestamps  =our_result_weighted{1,1};
rw_tx          =our_result_weighted{1,2};
rw_ty          =our_result_weighted{1,3};
rw_tz          =our_result_weighted{1,4};
rw_qx          =our_result_weighted{1,5};
rw_qy          =our_result_weighted{1,6};
rw_qz          =our_result_weighted{1,7};
rw_qw          =our_result_weighted{1,8};

gt_t = [gt_tx, gt_ty, gt_tz];
res_t = [res_tx, res_ty, res_tz];
rw_t = [rw_tx, rw_ty, rw_tz];

q_gt1 = [-gt_qw(1) gt_qx(1) gt_qy(1) gt_qz(1)];
n2 = quatrotate(q_gt1, res_t);
n3 = quatrotate(q_gt1, rw_t);

transformed = [gt_tx(1)+n2(:,1),gt_ty(1)+n2(:,2),gt_tz(1)+n2(:,3)];
transformed2 = [gt_tx(1)+n3(:,1),gt_ty(1)+n3(:,2),gt_tz(1)+n3(:,3)];

%%
close all
plot3(gt_tx, gt_ty, gt_tz,'k','linewidth',0.5);
axis equal
hold on
xmax = max(gt_tx)+0.25;
ymax = max(gt_ty)+0.25;
zmax = max(gt_tz)+0.25;
xmin = min(gt_tx)-0.25;
ymin = min(gt_ty)-0.25;
zmin = min(gt_tz)-0.25;

plot3(transformed(:,1),transformed(:,2),transformed(:,3),'r','linewidth',1);
plot3(transformed2(:,1),transformed2(:,2),transformed2(:,3),'g','linewidth',2);
j = waitforbuttonpress
gi = 2;
hold off
for i = 2:size(res_tx,1)
    
    while(gt_timestamps(gi) < res_timestamps(i))
        plot3(gt_tx(gi-1:gi), gt_ty(gi-1:gi), gt_tz(gi-1:gi),'k','linewidth',0.5);
        gi = gi+1; 
        axis equal
        axis([ xmin xmax ymin ymax zmin zmax]);
        
    end
    hold on
    plot3(transformed(i-1:i,1),transformed(i-1:i,2),transformed(i-1:i,3),'r','linewidth',1);
    plot3(transformed2(i-1:i,1),transformed2(i-1:i,2),transformed2(i-1:i,3),'g','linewidth',2);
    
    if mod(i,20) == 0
        v = [transformed2(i,:) ; transformed(i,:)];
        u = [gt_t(gi,:) ; transformed2(i,:)];
        plot3(v(:,1),v(:,2),v(:,3),'y')
        plot3(u(:,1),u(:,2),u(:,3),'y')
    end
    if mod(i,5) == 0
        j = waitforbuttonpress;
    end
end