
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import py4j.GatewayServer;

import com.codingame.gameengine.runner.SoloGameRunner;
import com.codingame.gameengine.runner.dto.GameResult;

public class SkeletonMain {
	public static Map<String, List<String>> simulate(int i, String p) {
        // Uncomment this section and comment the other one to create a Solo Game
        /* Solo Game */
        SoloGameRunner gameRunner = new SoloGameRunner();
        // Sets the player
        // Sets a test case
        gameRunner.setTestCase("test"+i+".json");

        // Another way to add a player for python
        gameRunner.setAgent("/Users/julian/opt/miniconda3/bin/python3 /Users/julian/Downloads/codigo_4/agent"+p);

        // Start the game server
        //gameRunner.start();
        // Simulate
        GameResult gr = gameRunner.simulate();
        Map<String, List<String>> errs = gr.errors;
        return errs;
	}
	public static void start() {
        // Uncomment this section and comment the other one to create a Solo Game
        /* Solo Game */
        SoloGameRunner gameRunner = new SoloGameRunner();
        // Sets the player
        // Sets a test case
        gameRunner.setTestCase("test4.json");

        // Another way to add a player for python
        gameRunner.setAgent("/usr/bin/python3 /home/miguel/Escritorio/age/best_agents/best_agent_all.py");
        // Start the game server
        //gameRunner.start();
        // Simulate
        gameRunner.start();
        
	}
    public static void main(String[] args) {
    	SkeletonMain app = new SkeletonMain();
        //app is now the gateway.entry_point
        GatewayServer server = new GatewayServer(app);
        server.start();
        //start();

    }
}
