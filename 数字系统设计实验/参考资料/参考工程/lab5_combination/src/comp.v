module comp(a, b, agb, aeb, alb);
    parameter n=1;

    // Inputs
    input [n-1:0] a,b;

    // Outputs
    output agb, aeb, alb;

    // Assign value
    assign agb = (a > b);
    assign aeb = (a == b);
    assign alb = (a < b);
endmodule
