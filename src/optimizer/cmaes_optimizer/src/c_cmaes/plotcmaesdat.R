
# This is an R script.  R is a freely available high level computer
# language, similar to S/S-plus. Its main area of application is
# statical computing and graphics. For information about R look at
# http://cran.r-project.org If you have installed and started R change
# to the directory where the files plotcmaesdat.R, rescmaes.dat, and
# xcmaes.dat can be found, e.g. type:
#   setwd("C:/path_of_dir/here_is_the_right_dir")
# Then type
#   source("plotcmaesdat.R")
# Remark that the absolute value of the fitness function value is
# plotted.

#--------------------------------------------------------------------

{
# Application oriented plot

  filename <- "rescmaes.dat"
  xfilename <- "xcmaes.dat";
  flg.plotx <- 1; 

  varnames <- c("eval", "fitness", "sigma", "maxstddev", "minstddev",
                "axisratio", "minDii")
  outy <- read.table(filename);
  attr(outy, "names")[1:7] <- varnames;
  len <- dim(outy)[2];

  minfit <- min(outy$fitness, na.rm=TRUE)
  min2fit <- min(outy$fitness[outy$fitness>minfit], na.rm=TRUE)
  diffitness <- outy$fitness - minfit 
  diffitness[diffitness < 1e-99] <- NA
  outy$fitness <- abs(outy$fitness)
  
  xlimits  <- c(0,max(outy[,1], finite=TRUE))
  ylimits <- range(c(abs(outy$fitness)+1e-99,
                     outy[,4:len], diffitness), finite=TRUE);
  
  xlabel <- "function evaluations"
  ylabel <- paste("func val minus ", format(minfit),
               " (>=", format(min2fit - minfit, digits=3), ",", 
               " cyan)", sep="")
  upper.label <- paste("func val (blue), axis ratio (red),",
                       "min&max std dev (black), main axes (green)")
  windows(7,6);
  par(new=FALSE); # few(diag(D))
  matplot(outy$eval, outy[,8:len], log="y", type="l", lty = 1, col="green",
          xlim=xlimits, ylim=ylimits, main="", xlab="",ylab="");
  
  par(new=TRUE);  # min&max std dev
  matlines(outy$eval, outy[,4:5], log="y", lty = 1, col="black",
           xlim=xlimits, ylim=ylimits, main="", xlab="",ylab="");

  par(new=TRUE);  # axisratio
  plot(outy$eval, outy$axisratio, log="y", type="l",col="red",
       xlim=xlimits, ylim=ylimits, main="", xlab="",ylab="");
  
  par(new=TRUE);  # difference to best function value
  plot(outy$eval, diffitness, log="y", type="l",col="cyan", 
       xlim=xlimits, ylim=ylimits, main="", xlab="",ylab="");

  idx <- outy$fitness>1e-99
  par(new=TRUE);  # positive function values, points
  plot(outy$eval[idx],outy$fitness[idx], log="y", type="p", col="blue",
       xlim=xlimits, ylim=ylimits, main="", xlab="",ylab="");
  idx <- outy$fitness < -1e-99
  par(new=TRUE);  # negative function values, points
  plot(outy$eval[idx],-outy$fitness[idx],
       log="y", type="p", col="red",
       xlim=xlimits, ylim=ylimits, main="", xlab="",ylab="");
  par(cex.lab=0.9);
  par(new=TRUE);  # all function values, lines
  plot(outy$eval,abs(outy$fitness)+1e-99, log="y", type="l", col="blue",
       xlim=xlimits, ylim=ylimits, main="", xlab="",ylab="")
  title(filename, xlab=xlabel, ylab=ylabel)
  mtext(upper.label, side=3, cex=0.9)
  grid();
}
if (flg.plotx)
{
  x <- read.table(xfilename);
  lenx <- dim(x)[2];
  xrange <- range(x[,1], na.rm=TRUE)
  yrange <- range(x[,2:lenx], finite=TRUE)
  windows(3,3)
  matplot(x[,1], x[,2:lenx], type = "l", lty = 1, col=rainbow(lenx),
           xlim=xrange, ylim=yrange, main="",
           ylab="objective variables", xlab="function evaluations");
  title(xfilename)
  grid();
}
