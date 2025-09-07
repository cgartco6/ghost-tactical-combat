import UIKit
import SpriteKit
import GoogleMobileAds

class GameViewController: UIViewController {
    var gameScene: GameScene!
    var adManager: AdManager!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Initialize AdMob
        GADMobileAds.sharedInstance().start(completionHandler: nil)
        
        // Configure the view
        let skView = view as! SKView
        skView.showsFPS = true
        skView.showsNodeCount = true
        
        // Create and configure the scene
        gameScene = GameScene(size: skView.bounds.size)
        gameScene.scaleMode = .aspectFill
        
        // Present the scene
        skView.presentScene(gameScene)
        
        // Initialize ad manager
        adManager = AdManager(viewController: self)
    }
    
    override var shouldAutorotate: Bool {
        return true
    }
    
    override var supportedInterfaceOrientations: UIInterfaceOrientationMask {
        return .landscape
    }
    
    override var prefersStatusBarHidden: Bool {
        return true
    }
    
    func showAd(completion: @escaping (Bool) -> Void) {
        adManager.showRewardedAd(completion: completion)
    }
}
