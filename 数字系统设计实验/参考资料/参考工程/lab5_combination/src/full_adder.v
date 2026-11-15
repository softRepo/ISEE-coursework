module full_adder(a, b, s, ci, co);
    // Inputs
    input a, b, ci;

    // Outputs
    output s, co;

    // Assign value
    assign s = a ^ b ^ ci;
    assign co = (a & b) | (ci & b) | (ci & a);
endmodule
