module synchronizer(in, clk, out);
    input in, clk;
    output reg out;
    reg connection;

    always @(posedge clk) begin
        connection = in;
    end

    always @(posedge clk) begin
        out = connection;
    end
endmodule
