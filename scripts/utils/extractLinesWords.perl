#!/usr/bin/perl

###############################################################################
# Purpose: Extracting a subset of lines from an embedding file. The especific 
#          words to extract are listed in a text file
# Author:  Cristina
# Date:    16/08/2018
###############################################################################

use strict;
use warnings;

#binmode(STDIN, ":utf8");
#binmode(STDOUT, ":utf8");


# Command line arguments
my $num_args = $#ARGV + 1;
if ($num_args != 2) {
    print "\nUsage: $0 <listLinesFile> <fullFile>  >  extractedLines \n";
    exit;
}

my $lines2extract=$ARGV[0];
my $completeFile=$ARGV[1];


# Load lines to extract
open LINES, "< $lines2extract" or die "could not open $lines2extract\n";
my $keyRef;
while (<LINES>) {
   chomp;
   $keyRef->{$_} = 1;
}
close LINES;

# Traverse input file and extract selected lines
open FILE2, "< $completeFile" or die "could not open $completeFile\n";
my $i=1;
while (<FILE2>) {
    chomp;
    my $line = $_;
    my $word = substr( $line, 0, index( $line, ' ' ) );
    if (defined $keyRef->{$word}) {
        print STDOUT "$line\n";
    }
    $i++;
}
close FILE2;
