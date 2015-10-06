%
% plots output files from example.c in matlab
%
  filename = 'rescmaes.dat';
  xfilename = 'xcmaes.dat'; 
  flgplotx = 1; 

  outy = load(filename);
  minfit = min(outy(:,2)); min2fit = min(outy(find(outy(:,2)>minfit),2));
  diffit = outy(:,2) - minfit; 
  diffit(diffit<1e-99) = NaN; 

  disp(outy(end,2));
  figure(326); hold off; 
  semilogy(outy(:,1), outy(:,8:end), '-g'); hold on; % few(diag(D))
  semilogy(outy(:,1), outy(:,4:5), '-k'); hold on;   % min&max std dev
  semilogy(outy(:,1), outy(:,6), '-r'); hold on;     % axis ratio
  semilogy(outy(:,1), diffit, '-c'); hold on; % difference to best func. val.
  idx = find(outy(:,2)>1e-99);
  semilogy(outy(idx,1), abs(outy(idx,2)), '.b'); hold on; % positive values
  idx = find(outy(:,2) < -1e-99);
  semilogy(outy(idx,1), abs(outy(idx,2)), '.r'); hold on; % negative values
  semilogy(outy(:,1), abs(outy(:,2))+1e-99, '-b'); hold on;
  title(['abs(func val) (blue), Axis Ratio (red), Main Axes (green), ' ...
	 'Max&Min Std Dev (black)'] ); 
  xlabel('function evaluations');
  ylabel([filename ': func val minus ' num2str(minfit) ' (>= ' ...
	  num2str(min2fit-minfit) ', cyan)']);
  grid on; zoom on;

  if flgplotx
    figure(327); hold off; 
    prop = get(326); pos = prop.Position; 
    pos(1:2) = 0.9*pos(1:2);  pos(3:4) = 0.4*pos(3:4);
    set(327, 'Position', pos);
    outx = load(xfilename);
    plot(outx(:,1), outx(:,2:end), '-');
    title(xfilename);
    xlabel('function evaluations');
    ylabel('objective variables');
    grid on; zoom on;
  end

